"""
Human-readable labels for variable names used across analysis scripts.
TMPT and ETF abbreviations are kept as-is.
"""

# Regression variables
VAR_LABELS = {
    # Intercept
    "const":         "Intercept",

    # Autoregressive price terms
    "X_AR_ICLN":     "AR(ICLN)",
    "X_AR_IXC":      "AR(IXC)",
    "X_AR_VDE":      "AR(VDE)",
    "X_AR_XLE":      "AR(XLE)",
    "X_AR_XOP":      "AR(XOP)",

    # Energy news density
    "X_ED":          "Energy News Density",

    # Event variables
    "X_E_ARE_SAU":   "UAE & Saudi Arabia (COP28)",
    "X_E_ESP":       "Spain (Power Outage)",
    "X_E_BRA":       "Brazil (COP30)",
    "X_E_GEO":       "Georgia (Gas Dispute)",
    "X_E_DEU":       "Germany (Nord Stream)",
    "X_E_IRN_ISR":   "Iran–Israel (Gulf Conflict)",
    "X_E_UKR":       "Ukraine (Invasion)",
}

# ETF full names (abbreviations kept in column headers)
ETF_LABELS = {
    "ICLN": "ICLN",
    "IXC":  "IXC",
    "VDE":  "VDE",
    "XLE":  "XLE",
    "XOP":  "XOP",
}

# Outcome / descriptive variables
SERIES_LABELS = {
    "energy_price_proportion": "Energy Price Article Share (%)",
    "grand_mean":              "Grand Mean TMPT Score",
    "N_rel":                   "Energy Price Articles",
    "N_total":                 "Total Articles",
    "N_irrel":                 "Irrelevant Articles",
    "EP Share (%)":            "Energy Price Share (%)",
    "mean_rate":               "Mean Mention Rate",
    "zscore":                  "Z-score",
}


def label(name: str) -> str:
    """Return the human-readable label for a variable name, or the name itself if not found."""
    return VAR_LABELS.get(name) or SERIES_LABELS.get(name) or ETF_LABELS.get(name) or name
