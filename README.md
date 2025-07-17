# PopSyCLE Table Manipulation

This repository contains scripts to process raw simulation outputs from the Population Synthesis for Compact object Lensing Events (PopSyCLE) pipeline into a user-friendly, analysis-ready catalog of binary-source microlensing events.

## Features

- Load and process FITS files using Astropy and Pandas
- Filter and manipulate large astronomical datasets
- Calculate derived physical quantities (e.g., distances, angular separations)
- Export results to CSV for further analysis

## Project Structure

The repository is organized as follows:

```
project/
├── data/                # Directory for input and output data (should be in .gitignore)
│   ├── all_fields_Mrun_EWS_w_comp_params.fits  (Input)
│   ├── t_comp_rb_no_cuts_all_fields.fits       (Unused)
│   ├── t_lightcurves_no_cuts_all_fields.fits   (Unused)
│   ├── column_names.txt                        (Outout) # Column names for the all_fields_Mrun_EWS_w_comp_params.fits table
│   └── popsycle_table_with_calc.csv            (Output)
├── make_database.ipynb  # Jupyter Notebook to process data
├── environment.yml      # Conda environment definition
└── README.md            # This file

```

## Requirements

The primary dependencies are Python 3.11 and the packages listed in the environment file.

- [Astropy](https://www.astropy.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)

You can install the complete environment using Conda:

```bash
conda env create -f environment.yml
conda activate popsycle
````

## Workflow

1.  **Setup:** Clone this repository and activate the Conda environment as described above.

2.  **Data Placement:** Download the necessary PopSyCLE data (`all_fields_Mrun_EWS_w_comp_params.fits`) and place it inside the `data/` directory.

3.  **Execution:** Open and run the `make_database.ipynb` Jupyter Notebook from top to bottom. This will perform all filtering and calculations.

4.  **Output:** The notebook will generate `popsycle_table_with_calc.csv` in the `data/` directory. This file contains the final, clean catalog of binary-source events.

## Using the Output Data

The final output table contains the following key columns with their corresponding units:

| Your Column Name      | Original Column(s)        | Units                                         |
| :-------------------- | :------------------------ | :-------------------------------------------- |
| `M_L`                 | `mass_L`                  | Solar Masses ($M_\odot$)                      |
| `D_L_kpc`             | `px_L`, `py_L`, `pz_L`    | Kiloparsecs (kpc)                             |
| `D_S_kpc`             | `px_S`, `py_S`, `pz_S`    | Kiloparsecs (kpc)                             |
| `mu_rel_mas_yr`       | `mu_rel`                  | Milliarcseconds per year (mas/yr)             |
| `theta_E_mas`         | `theta_E`                 | Milliarcseconds (mas)                         |
| `u0`                  | `u0`                      | $\theta_E$                                    |
| `binary_sep_au`       | `comp_S_sep`              | Projected separation (AU)                     |
| `binary_log_a_au`     | `comp_S_log_a`            | log10(semi-major axis/AU)                     |
| `binary_a_au`         | `comp_S_log_a`            | Semi-major axis (AU)                          |
| `binary_alpha_deg`    | `comp_S_alpha`            | Orientation angle (deg)                       |
| `binary_phi_deg`      | `comp_S_phi`              | Phase angle (deg)                             |
| `binary_sep_arcsec`   | `comp_S_sep`, `D_S`       | Projected separation (arcsec)                 |
| `gal_lon_S_deg`       | `glon_S`                  | Galactic longitude (deg)                      |
| `gal_lat_S_deg`       | `glat_S`                  | Galactic latitude (deg)                       |
| `I_S`                 | `ubv_I_S`                 | Source I-band magnitude                       |
| `I_S2`                | `comp_S_m_ubv_I`          | Source companion I-band magnitude             |
| `mu_b_L`              | `mu_b_L`                  | Lens proper motion in b direction (mas/yr)    |
| `mu_lcosb_L`          | `mu_lcosb_L`              | Lens proper motion in l*cos(b) (mas/yr)       |
| `mu_b_S`              | `mu_b_S`                  | Source proper motion in b direction (mas/yr)  |
| `mu_lcosb_S`          | `mu_lcosb_S`              | Source proper motion in l*cos(b) (mas/yr)     |

You can easily load and begin working with this table using Pandas:

```python
import pandas as pd

# Load the analysis-ready catalog
df = pd.read_csv('data/popsycle_table_with_calc.csv')

# Display the first few events
print(df.head())
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Data

The input data for this project is sourced from the PopSyCLE simulation pipeline. The license in this repository (MIT License) applies **only** to the source code (`.ipynb`, `.py` files) and not to the data itself. Use and distribution of the PopSyCLE data and any derivative products are subject to the terms and policies provided by the original authors. Please consult the [PopSyCLE website](https://w.astro.berkeley.edu/popsycle/) for their specific data use policy.

## Acknowledgments

This work utilizes data generated by the PopSyCLE pipeline. We acknowledge Natasha Abrams for providing the pre-processed data table used in this analysis.

