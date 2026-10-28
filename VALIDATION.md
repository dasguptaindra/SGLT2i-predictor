# Validation — 28 October 2026

- Tested in an isolated Windows Python 3.11.3 environment with Java 17.0.20.1 and the pinned requirements.
- `pip check` passed; Python 3.11/Linux x86_64 dependency resolution passed. Linux runtime and Streamlit Cloud deployment were not executed.
- PaDEL returned `minaaN` and `nHBAcc_Lipinski` using the selected descriptor families.
- Combined 13-feature vectors were finite and matched the actual model's feature order for `Clc1ncc(C(C)C)cc1`, aspirin (`CC(=O)OC1=CC=CC=C1C(=O)O`), ethanol (`CCO`) and acetamide (`NC(=O)C`).
- Missing Mordred values were zero-filled using the user-confirmed training convention. This included absent E-state types and undefined MDEC-33 in small molecules.
- The actual model loaded and predicted for all four vectors. SHAP class-1 additivity matched predicted class-1 probability.
- Streamlit AppTest passed startup, Predict button interaction, descriptor calculation, prediction, two result metrics and SHAP plotting for aspirin and `Clc1ncc(C(C)C)cc1`.
- Invalid SMILES was rejected without an unhandled exception. Ketcher's browser drawing interface was not manually exercised.
- The delivered model is byte-for-byte identical to the supplied model.

PaDEL's Lipinski acceptor count is not universally interchangeable with Mordred/RDKit's `nHBAcc`: acetamide returned 2 for PaDEL `nHBAcc_Lipinski` and 1 for RDKit `CalcNumHBA`. The app uses the direct PaDEL output.

These are software execution checks, not a new accuracy evaluation. The explicit source mapping is PaDEL for minaaN and nHBAcc_Lipinski, Mordred for the other 11 names. Exact equivalence to the historical training vectors still requires those vectors and the original calculator settings.
