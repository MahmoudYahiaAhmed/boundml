use boundml_artifact::Package;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: power-app <model.boundml> [name=value ...]")?;
    let package = Arc::new(Package::load(path.as_ref())?); // checked: hashes, version, graph, memory plan
    let mut predictor = package.predictor(1)?;

    // Feature values by name; an empty value means "missing".
    let pairs: Vec<(String, Option<f64>)> = args
        .map(|a| {
            let (k, v) = a
                .split_once('=')
                .ok_or(format!("expected name=value, got {a}"))?;
            Ok((
                k.to_string(),
                if v.is_empty() { None } else { Some(v.parse()?) },
            ))
        })
        .collect::<Result<_, Box<dyn std::error::Error>>>()?;
    let mut raw = vec![None; predictor.n_features()];
    predictor.fill_named(pairs.iter().map(|(k, v)| (k.as_str(), *v)), &mut raw, 0)?;

    let mut out = vec![0.0f32; predictor.output_cols()];
    predictor.predict_raw(&raw, 1, &mut out)?; // fitted preprocessing + model
    for (name, value) in package.manifest().output_names().iter().zip(&out) {
        println!("{name} = {value:.3}");
    }
    Ok(())
}
