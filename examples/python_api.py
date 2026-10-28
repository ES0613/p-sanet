"""Minimal Python API example. Download SMI-TED-Light before running."""

from p_sanet import predict_range

result = predict_range("CCO", 280.0, 320.0, n_points=5)
result.write("example_output")
print(result.ensemble_summary.to_string(index=False))

heat = predict_range("CCO", 280.0, 320.0, n_points=5,
                     properties=["heat_of_vaporization"])
print(heat.thermodynamics_summary.to_string(index=False))

critical = predict_range("CCO", 280.0, 320.0, n_points=5,
                         properties=["critical_properties"])
print(critical.critical_properties_summary.to_string(index=False))
