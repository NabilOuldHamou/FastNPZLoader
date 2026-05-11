use pyo3::prelude::*;

#[pymodule]
mod fast_npz {
    use indicatif::{ProgressBar, ProgressStyle};
    use ndarray_npy::ReadNpyExt;
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use rayon::prelude::*;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{Cursor, Read};

    #[derive(Debug)]
    enum NpyData {
        Float64(Vec<f64>),
        Float32(Vec<f32>),
        Strings(Vec<String>),
    }

    impl<'py> IntoPyObject<'py> for NpyData {
        type Target = PyAny;
        type Output = Bound<'py, PyAny>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            match self {
                NpyData::Float64(v) => v.into_pyobject(py).map(|b| b.into_any()),
                NpyData::Float32(v) => v.into_pyobject(py).map(|b| b.into_any()),
                NpyData::Strings(v) => v.into_pyobject(py).map(|b| b.into_any()),
            }
        }
    }

    /// Parse the NPY magic + header once, return (type_char, elem_size, little_endian, data_offset).
    fn peek_npy_header(bytes: &[u8]) -> Option<(char, usize, bool, usize)> {
        if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
            return None;
        }
        let major = bytes[6];
        let (header_len, data_offset) = if major == 1 {
            (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
        } else {
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            )
        };
        let header = std::str::from_utf8(&bytes[data_offset..data_offset + header_len]).ok()?;

        let di = header.find("'descr':")?.saturating_add("'descr':".len());
        let rest = header[di..].trim_start();
        let quote = rest.chars().next()?;
        let inner = &rest[1..];
        let descr = &inner[..inner.find(quote)?];

        let mut chars = descr.chars();
        let endian = chars.next()?;
        let type_char = chars.next()?;
        let size: usize = chars.collect::<String>().parse().ok()?;

        Some((type_char, size, endian != '>', data_offset + header_len))
    }

    fn parse_byte_strings(data: &[u8], elem_size: usize) -> Vec<String> {
        data.chunks(elem_size)
            .map(|c| {
                let end = c.iter().position(|&b| b == 0).unwrap_or(c.len());
                String::from_utf8_lossy(&c[..end]).into_owned()
            })
            .collect()
    }

    fn parse_unicode_strings(data: &[u8], char_count: usize, little_endian: bool) -> Vec<String> {
        let elem_bytes = char_count * 4;
        data.chunks(elem_bytes)
            .map(|chunk| {
                let mut s = String::with_capacity(char_count);
                for c in chunk.chunks(4).filter(|c| c.len() == 4) {
                    let cp = if little_endian {
                        u32::from_le_bytes([c[0], c[1], c[2], c[3]])
                    } else {
                        u32::from_be_bytes([c[0], c[1], c[2], c[3]])
                    };
                    if cp == 0 {
                        break;
                    }
                    if let Some(ch) = char::from_u32(cp) {
                        s.push(ch);
                    }
                }
                s
            })
            .collect()
    }

    /// Dispatch to the right parser using a single header parse.
    fn parse_npy_entry(bytes: &[u8]) -> Option<NpyData> {
        let (type_char, size, little_endian, data_offset) = peek_npy_header(bytes)?;
        match (type_char, size) {
            ('f', 8) => ndarray::ArrayD::<f64>::read_npy(Cursor::new(bytes))
                .ok()
                .map(|a| NpyData::Float64(a.into_raw_vec_and_offset().0)),
            ('f', 4) => ndarray::ArrayD::<f32>::read_npy(Cursor::new(bytes))
                .ok()
                .map(|a| NpyData::Float32(a.into_raw_vec_and_offset().0)),
            ('S', n) => Some(NpyData::Strings(parse_byte_strings(
                &bytes[data_offset..],
                n,
            ))),
            ('U', n) => Some(NpyData::Strings(parse_unicode_strings(
                &bytes[data_offset..],
                n,
                little_endian,
            ))),
            _ => None,
        }
    }

    fn load_single_npz(path: &str) -> Result<HashMap<String, NpyData>, String> {
        // Phase 1 (sequential): open zip once, decompress every entry into memory.
        let file = File::open(path).map_err(|e| e.to_string())?;
        let mut archive = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;

        let mut entries: Vec<(String, Vec<u8>)> = Vec::with_capacity(archive.len());
        for i in 0..archive.len() {
            let mut entry = archive.by_index(i).map_err(|e| e.to_string())?;
            if !entry.name().ends_with(".npy") {
                continue;
            }
            let name = entry.name().trim_end_matches(".npy").to_string();
            let mut buf = Vec::with_capacity(entry.size() as usize);
            entry.read_to_end(&mut buf).map_err(|e| e.to_string())?;
            entries.push((name, buf));
        }

        // Phase 2 (parallel): parse NPY headers and decode arrays across cores.
        Ok(entries
            .into_par_iter()
            .filter_map(|(name, buf)| parse_npy_entry(&buf).map(|data| (name, data)))
            .collect())
    }

    #[pyfunction]
    fn load_from_directory(py: Python<'_>, directory: String) -> PyResult<Bound<'_, PyDict>> {
        let paths: Vec<String> = std::fs::read_dir(&directory)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path().to_string_lossy().to_string())
            .filter(|p| p.ends_with(".npz"))
            .collect();

        let pb = ProgressBar::new(paths.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} files | {elapsed} elapsed | eta {eta}")
                .unwrap()
                .progress_chars("=> "),
        );

        let results: Vec<(String, Result<HashMap<String, NpyData>, String>)> = paths
            .into_par_iter()
            .map(|path| {
                let data = load_single_npz(&path);
                pb.inc(1);
                (path, data)
            })
            .collect();

        pb.finish_with_message("done");

        let outer = PyDict::new(py);
        for (path, arrays_result) in results {
            let arrays = arrays_result.map_err(PyRuntimeError::new_err)?;
            let inner = PyDict::new(py);
            for (name, data) in arrays {
                inner.set_item(name, data.into_pyobject(py)?)?;
            }
            outer.set_item(path, inner)?;
        }
        Ok(outer)
    }
}
