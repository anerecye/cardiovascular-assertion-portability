def normalize_label(value):
    return " ".join(str(value or "").strip().lower().replace("|", ";").split())


def infer_environment(condition, domain):
    text = normalize_label(condition)
    domain = str(domain or "").lower()
    if not text or "not provided" in text:
        return "unknown"
    if domain == "inherited_arrhythmia":
        if "brugada" in text:
            return "Brugada syndrome"
        if "long qt" in text or "lqts" in text:
            return "long QT syndrome"
        if "catecholaminergic" in text or "cpvt" in text:
            return "CPVT"
        return "arrhythmia"
    if domain == "cardiomyopathy":
        if "hypertrophic" in text or "hcm" in text:
            return "HCM"
        if "dilated" in text or "dcm" in text:
            return "DCM"
        if "arrhythmogenic" in text:
            return "arrhythmogenic cardiomyopathy"
        return "cardiomyopathy"
    if domain == "hereditary_cancer":
        if "breast" in text or "ovarian" in text:
            return "breast/ovarian cancer predisposition"
        if "lynch" in text or "colon" in text or "colorectal" in text:
            return "colorectal cancer predisposition"
        return "hereditary cancer predisposition"
    return "unknown"
