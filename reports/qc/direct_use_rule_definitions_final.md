# Final direct-use rule definitions

CAB-Strict deterministic direct use is allowed only when source identity is accepted, meaning identity is accepted, phenotype-domain discordance is false, the disease architecture is phenotype_anchored_monogenic or syndrome_anchored_self_loop, and the baseline environment is specific/concordant.

CAB-Strict blocks nonspecific_underresolved, modifier_penetrance_boundary, structural_functional_overlap, syndrome_organ_boundary unless explicitly self-loop concordant, trigger_dependent_latent, pleiotropic_collision, genotype_first_absent_phenotype, source-unmatched, meaning-rejected, and phenotype-domain-discordant assertions.

CAB-Balanced may be more permissive than Strict but still blocks source_match_accepted=False, meaning_match_accepted=False, phenotype_domain_discordance_flag=True, nonspecific_underresolved unless repaired/specific context exists, genotype_first_absent_phenotype, explicit no-deterministic-reuse states, and nonspecific baseline environments.
