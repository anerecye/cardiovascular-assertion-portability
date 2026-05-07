import re

def normalize_label(label):
    s = str(label or "").lower()
    s = re.sub(r"[_/|]+", " ", s)
    s = re.sub(r"[^a-z0-9+ -]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def infer_environment(label, domain):
    s = normalize_label(label)
    if domain == "inherited_arrhythmia":
        if "long qt" in s or "lqts" in s: return "phenotype_anchored_lqts"
        if "brugada" in s: return "brugada_syndrome"
        if "cpvt" in s or "catecholaminergic" in s: return "provocation_dependent_arrhythmia"
        if "sudden" in s or "sads" in s or "death" in s: return "postmortem_or_sads"
        return "arrhythmia_other_or_unspecified"
    if domain == "cardiomyopathy":
        if "hypertrophic" in s or "hcm" in s: return "hypertrophic_cardiomyopathy"
        if "dilated" in s or "dcm" in s: return "dilated_cardiomyopathy"
        if "arrhythmogenic" in s or "arvc" in s: return "arrhythmogenic_cardiomyopathy"
        return "cardiomyopathy_other_or_unspecified"
    if domain == "hereditary_cancer":
        if "breast" in s or "ovarian" in s: return "breast_ovarian_cancer"
        if "lynch" in s or "colorectal" in s: return "lynch_or_colorectal"
        if "polyposis" in s: return "polyposis"
        return "cancer_predisposition_other_or_unspecified"
    return "unknown_environment"
