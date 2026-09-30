"""Greek and English text for the web interface, and the one-line help behind every field.

Two dictionaries, one shape. `T` holds what a control is called, `H` holds what it does. Both map
a key to `(greek, english)`, in that order, because Greek is the default here.

**Why the help text lives in code rather than in the templates.** A field whose meaning is written
beside it in HTML is explained once, in one language, and nothing notices when the field changes.
Keeping the text here means `tests/test_i18n.py` can walk the templates, collect every `name=` it
finds, and fail when one of them has no entry. A new control therefore cannot reach a user
unexplained, which is the same rule the manuscript's guard applies to its own numbers.

**The house rule for a help line.** Say what the control does in one sentence a person who has
never read a game-theory paper can follow, then, where it matters, what changes if they move it.
Name the trade-off rather than the mechanism. "Learns faster but forgets what it learned" beats
"step size of the temporal-difference update", and the second one is already in the docstrings for
anyone who wants it.

**What is deliberately left in English**, under the choice recorded for this interface: names that
are terms of art or code identifiers, since translating them cuts the reader off from the paper,
the repository and the literature. `Q-learning`, `DQN`, `MPCR`, `Nash`, `public_goods`,
`informed-v1` and the metric keys stay as they are, and the Greek sentence explains them instead.
"""

from __future__ import annotations

from typing import Optional

#: Language codes this interface serves, the first being the default.
LANGS = ("el", "en")
DEFAULT_LANG = "el"

#: Labels that are a canonical name rather than a description: a parameter as the code and the
#: ledger spell it, a measure as the papers name it, an architecture as its literature calls it.
#: They read the same in both languages on purpose, so that what is on screen, what is in the
#: stored record and what is in the paper are one string. Translating them invented terms nobody
#: uses; the Greek belongs in the badge, where it explains instead of renaming.
#: Every `(greek, english)` pair below is a visible string. Keys are grouped by where they appear.
T: dict[str, tuple[str, str]] = {
    # ---- chrome -------------------------------------------------------------------------
    "app.tagline":        ("Αρχιτεκτονικές που παίζουν επαναλαμβανόμενα παιχνίδια",
                           "Architectures that play repeated games"),
    "field.genome_origin": ("Από πού προκύπτει αυτό το κύκλωμα", "Where this circuit comes from"),
    "opt.genome_random":   ("Καινούργιο τυχαίο", "A fresh random one"),
    "opt.genome_cid":      ("Από κωδικό εξελιγμένου", "From an evolved circuit's identifier"),
    "nav.group.run":       ("Δοκιμή", "Run"),
    "nav.group.results":   ("Αποτελέσματα", "Results"),
    "nav.group.synthesis": ("Μεγάλη εικόνα", "Big picture"),
    "nav.match":          ("Παρτίδα", "Match"),
    "nav.evolve":         ("Εξέλιξη", "Evolutionary"),
    "nav.analytics":      ("Αναλύσεις", "Analytics"),
    "nav.sweep":          ("Σάρωση", "Sweep"),
    "nav.coverage":       ("Κάλυψη", "Coverage"),
    "nav.meta":           ("Μετα-ανάλυση", "Meta-analysis"),
    "nav.spacemap":       ("Χάρτης χώρου", "Space map"),
    "lang.switch":        ("Language: English", "Γλώσσα: Ελληνικά"),
    # Both stay in English in both languages: they are the names of the two skins, and a reader
    # who sees "Gamified" here and "gamified" in the paper is looking at the same thing.
    "theme.gamified":     ("Gamified", "Gamified"),
    "theme.scientific":   ("Scientific", "Scientific"),

    # ---- the match form -----------------------------------------------------------------
    "sec.game":           ("Το παιχνίδι", "The game"),
    "sec.roster":         ("Ποιοι παίζουν", "Who plays"),
    "sec.options":        ("Τι θα υπολογιστεί", "What gets computed"),
    "sec.batch":          ("Πολλές παρτίδες μαζί", "Many matches at once"),
    "field.game_kind":    ("game", "game"),
    "field.rounds":       ("rounds", "rounds"),
    "field.mpcr":         ("MPCR", "MPCR"),
    "field.seed":         ("seed", "seed"),
    "field.n_agents":     ("n_agents", "n_agents"),
    "field.cong_num_positions":      ("num_positions", "num_positions"),
    "field.cong_reward_rule":        ("reward_rule", "reward_rule"),
    "field.cong_memory_episodes":    ("memory_episodes", "memory_episodes"),
    "field.cong_episode_max_rounds": ("episode_max_rounds", "episode_max_rounds"),
    "field.cong_full_reward":        ("full_reward", "full_reward"),
    "field.cong_collapse_at_full":   ("collapse_at_full", "collapse_at_full"),
    "field.compute_nash":  ("Υπολογισμός ισορροπίας Nash", "Compute Nash equilibrium"),
    "field.compute_phi":   ("Υπολογισμός Φ (αιτιακή αυτονομία)", "Compute Φ (causal autonomy)"),
    "field.show_metrics":  ("Ποια μέτρα να δείξω", "Measures to show"),
    "field.log_lines":     ("Γραμμές αφήγησης", "Log lines"),
    "field.record_repo":   ("Καταγραφή στο αρχείο πειραμάτων", "Record into the ledger"),
    "field.batch_mode":    ("Τρέξε μια σειρά τιμών", "Run a series of values"),
    "field.batch_param":   ("Ποια παράμετρος αλλάζει", "Which parameter varies"),
    "field.batch_values":  ("Τιμές", "Values"),
    "field.wiz_rounds":    ("Γύροι", "Rounds"),
    "field.wiz_mix":       ("Έτοιμη σύνθεση παικτών", "Preset roster"),

    # ---- agents and their knobs ---------------------------------------------------------
    "agent.qlearning":    ("Q-learning", "Q-learning"),
    "agent.dqn":          ("Deep Q-Network", "Deep Q-Network"),
    "agent.fep":          ("FEP / Ενεργητική συμπερασματολογία", "FEP / Active Inference"),
    "agent.markov_brain": ("Markov-brain (εξελικτικό)", "Markov-brain (evolutionary)"),
    "agent.classic":      ("Κλασικές στρατηγικές (σταθερές)", "Classic (fixed)"),
    "agent.llm":          ("Γλωσσικό μοντέλο (Ollama)", "LLM (Ollama)"),
    "param.alpha":            ("alpha (α)", "alpha (α)"),
    "param.gamma":            ("gamma (γ)", "gamma (γ)"),
    "param.epsilon_decay":    ("epsilon_decay", "epsilon_decay"),
    "param.epsilon_min":      ("epsilon_min", "epsilon_min"),
    "param.hidden":           ("hidden", "hidden"),
    "param.lr":               ("lr", "lr"),
    "param.buffer_size":      ("buffer_size", "buffer_size"),
    "param.batch_size":       ("batch_size", "batch_size"),
    "param.train_every":      ("train_every", "train_every"),
    "param.target_sync_every": ("target_sync_every", "target_sync_every"),
    "param.reciprocity":      ("reciprocity", "reciprocity"),
    "param.obs_noise":        ("obs_noise", "obs_noise"),
    "param.drift":            ("drift", "drift"),
    "param.precision":        ("precision", "precision"),
    "param.n_hidden":         ("n_hidden", "n_hidden"),
    "param.strategy":         ("strategy", "strategy"),
    "param.p":                ("p_cooperate", "p_cooperate"),
    "param.backend_name":     ("backend", "backend"),
    "param.model":            ("model", "model"),
    "param.profile":          ("profile", "profile"),
    "param.native_schema":    ("constrained", "constrained"),
    "param.history":          ("history", "history"),
    "param.persona":          ("persona", "persona"),
    "param.p_cooperate":      ("p_cooperate", "p_cooperate"),
    "param.llm_model":        ("model", "model"),
    "param.llm_constrained":  ("constrained", "constrained"),
    "param.llm_profile":      ("profile", "profile"),
    "param.genome_cid":       ("genome CID", "genome CID"),
    "param.epsilon_pair":     ("epsilon_decay / epsilon_min", "epsilon_decay / epsilon_min"),
    "field.row_kind":         ("Αρχιτεκτονική", "Architecture"),
    "field.row_count":        ("Πόσοι", "How many"),
    "btn.details":            ("Λεπτομέρειες", "Details"),
    "btn.advanced":           ("Προχωρημένες παράμετροι", "Advanced parameters"),
    # step headings and the wizard chrome
    "sec.game_h":     ("Το παιχνίδι", "Game"),
    "sec.roster_h":   ("Ποιοι παίζουν: πρόσθεσε έναν κάθε φορά, το σύνολο βγαίνει μόνο του",
                       "Roster: add agents one at a time, the total is computed for you"),
    "sec.metrics_h":  ("Τι μέτρα να δείξω", "Metrics: what to show"),
    "sec.record_h":   ("Καταγραφή", "Recording"),
    "sec.batch_h":    ("Πολλές παρτίδες μαζί", "Batch mode"),
    "sec.congestion": ("Οικογένεια congestion", "Congestion family"),
    "sec.roadmap":    ("Σχεδιάζονται", "Roadmap"),
    "btn.next":       ("Επόμενο", "Next"),
    "btn.back":       ("Πίσω", "Back"),
    "btn.run_it":     ("Τρέξ' το", "Run it"),
    "btn.next_roster":  ("Επόμενο: Ποιοι παίζουν", "Next: Roster"),
    "btn.next_metrics": ("Επόμενο: Μέτρα", "Next: Metrics"),
    "btn.next_record":  ("Επόμενο: Καταγραφή", "Next: Recording"),
    "btn.next_batch":   ("Επόμενο: Πολλές μαζί", "Next: Batch"),
    "btn.blank_row":    ("κενή σειρά", "blank row"),
    "preset.one_of_each":  ("Ένας από κάθε είδος", "One of each"),
    "preset.rl_showdown":  ("Αναμέτρηση RL", "RL showdown"),
    "preset.evo_vs_online": ("Εξελικτικό εναντίον online", "Evolutionary vs online"),
    "field.compute_nash_l": ("Ισορροπία Nash (pygambit, μόνο για n έως 10)",
                            "Nash equilibrium (pygambit, n<=10 only)"),
    "field.compute_phi_l":  ("Φ / αιτιακή αυτονομία (μόνο αν παίζει markov_brain, αργό ~15 δευτ.)",
                             "Phi / causal autonomy (only with a markov_brain, slow ~15s)"),
    "field.record_repo_l":  ("Καταγραφή στο τοπικό αρχείο πειραμάτων (και στο Smart Filter)",
                             "Record to the repository-lite ledger (and the Smart Filter)"),
    "field.log_lines_l":    ("Πυκνότητα αφήγησης", "Narration density"),
    "field.batch_mode_l":   ("Τρέξε σειρά: άλλαξε ένα πεδίο σε πολλές τιμές, με χωριστή "
                             "καταγραφή για κάθε τρέξιμο",
                             "Run a batch: vary one field across several values, recording each "
                             "run separately"),
    "field.batch_param_l":  ("Πεδίο που σαρώνεται", "Field to sweep"),
    "field.batch_values_l": ("Τιμές", "Values"),
    "opt.rounds":           ("Γύροι", "Rounds"),
    "opt.seed":             ("Σπόρος", "Seed"),
    "opt.row0_count":       ("Πλήθος της πρώτης σειράς", "First roster row's count"),
    "opt.constrained_on":   ("1 — ο διακομιστής επιβάλλει τη μορφή",
                             "1 — the server enforces the format"),
    "opt.constrained_off":  ("0 — μόνο το prompt τη ζητά", "0 — only the prompt asks for it"),
    "group.none":    ("Χωρίς ομαδοποίηση, ένα τρέξιμο ανά γραμμή",
                      "No grouping, one run per row"),
    "group.design":  ("Ανά πείραμα", "By design"),
    "group.game":    ("Ανά παιχνίδι", "By game"),
    "group.roster":  ("Ανά σύνθεση παικτών", "By roster"),
    "field.group_by":  ("Ομαδοποίηση", "Group by"),
    "field.per_page":  ("Ανά σελίδα", "Per page"),
    "btn.prev_page":   ("Προηγούμενη", "Previous"),
    "btn.next_page":   ("Επόμενη", "Next"),
    "w.runs":          ("τρεξίματα", "runs"),
    "w.showing":       ("Δείχνω", "Showing"),
    "w.to":            ("έως", "to"),
    "w.page_of":       ("Σελίδα", "Page"),
    "w.slash_of":      ("από", "of"),
    "badge.planned":        ("σχεδιάζεται", "planned"),
    "btn.submit_run":       ("Τρέξε το πείραμα", "Run the experiment"),
    # page headings
    "h.sweep":        ("Άλλαξε ένα πράγμα, δες τα πάντα", "Move one thing, watch everything"),
    "h.sweep_prog":   ("Πρόοδος", "Progress"),
    "h.coverage":     ("Τι έχουμε τρέξει ως τώρα", "The board so far"),
    "h.correlations": ("Τι κινείται μαζί με τι", "What moves with what"),
    "h.evolve":       ("Εξέλιξη: γενετικός αλγόριθμος για Markov-brain",
                       "Evolutionary: Markov-brain genetic algorithm"),
    "h.ga_params":    ("Παράμετροι γενετικού αλγορίθμου", "Genetic algorithm parameters"),
    "h.selfplay":     ("Το παιχνίδι (αυτοπαιχνίδι)", "Game (self-play)"),
    "h.bakeoff":      ("Σύγκριση (τελική αναμέτρηση)", "Comparison (bake-off)"),
    "btn.update":     ("Ενημέρωση", "Update"),
    "btn.start_evolving": ("Ξεκίνα την εξέλιξη", "Start evolving"),
    "h.sources":      ("Πηγές", "Sources"),
    "h.pool":         ("Συνδυασμός πολλών τρεξιμάτων", "Pool across runs"),
    "h.designs":      ("Σχέδια πειραμάτων", "Designs"),
    "h.shared":       ("Σχέδια που έτρεξαν πάνω από μία εγκαταστάσεις",
                       "Designs more than one installation ran"),
    "h.filters":      ("Έξυπνα φίλτρα", "Smart filters"),
    "h.corr_chart":   ("Γράφημα συσχέτισης", "Correlation chart"),
    "h.matching":     ("Τρεξίματα που ταιριάζουν", "Matching runs"),
    "h.spacemap":     ("Χάρτης χώρου παραμέτρων", "Parameter-space map"),
    "h.coverage2":    ("Κάλυψη", "Coverage"),
    "btn.verify_import": ("Επαλήθευσε και εισήγαγε", "Verify and import"),
    "btn.export_bundle": ("Εξαγωγή αρχείου", "Export bundle"),
    "btn.pool":       ("Συνδύασε", "Pool"),
    "btn.add_filter": ("φίλτρο", "filter"),
    "btn.apply_filters": ("Εφάρμοσε τα φίλτρα", "Apply filters"),
    "btn.update_map": ("Ενημέρωσε τον χάρτη", "Update map"),
    "btn.run_missing_now": ("Τρέξε τώρα όσα κελιά λείπουν", "Run missing cells now"),
    "field.roster_has": ("Μόνο τρεξίματα που περιλαμβάνουν τουλάχιστον έναν από:",
                         "Only runs whose roster includes at least one of:"),
    "field.filter_field":    ("Πεδίο", "Field"),
    "field.filter_operator": ("Τελεστής", "Operator"),
    "field.filter_value":    ("Τιμή", "Value"),

    # ---- measures -----------------------------------------------------------------------
    "metric.cooperation_rate":     ("Cooperation rate", "Cooperation rate"),
    "metric.efficiency":           ("Efficiency", "Efficiency"),
    "metric.payoff_gini":          ("Payoff Gini", "Payoff Gini"),
    "metric.action_entropy_bits":  ("Action entropy (bits)", "Action entropy (bits)"),
    "metric.mutual_information_bits": ("Mutual information (bits)", "Mutual information (bits)"),
    "metric.transfer_entropy_bits": ("Transfer entropy (bits)", "Transfer entropy (bits)"),
    "metric.predictive_information_bits": ("Predictive information (bits)", "Predictive information (bits)"),
    "metric.coop_graph_weight":     ("Co-cooperation weight", "Co-cooperation weight"),
    "metric.coop_graph_clustering": ("Co-cooperation clustering", "Co-cooperation clustering"),
    "metric.influence_graph_density": ("Influence graph density", "Influence graph density"),
    "metric.phi":                   ("Φ (causal autonomy)", "Φ (causal autonomy)"),
    # ---- sweep --------------------------------------------------------------------------
    "field.axis":     ("Τι μεταβάλλεται", "Vary"),
    "field.values":   ("values", "values"),
    "field.seeds":    ("seeds", "seeds"),
    "field.workers":  ("workers", "workers"),
    "field.mix":      ("Σύνθεση παικτών", "Roster"),
    "field.agents":   ("n_agents", "n_agents"),
    "field.game":     ("game", "game"),
    # ---- coverage -----------------------------------------------------------------------
    "field.unit":     ("Ένα σημείο ανά", "One point per"),
    "field.family":   ("Ποια πεδία", "Fields"),
    "field.limit":    ("Το πολύ τόσα πεδία", "At most this many fields"),

    # ---- meta-analysis ------------------------------------------------------------------
    "field.metric":      ("Μέτρο", "Measure"),
    "field.only_local":  ("Μόνο δικά μου αποτελέσματα", "Only this installation's results"),
    "field.bundle":      ("Αρχείο αλυσίδας για εισαγωγή", "Import a bundle"),
    "field.trusted_key": ("Το δημόσιο κλειδί τους", "Their public key"),
    "field.label":       ("Ετικέτα", "Label"),
    "field.title":       ("Τίτλος", "Title"),
    "field.doi":         ("DOI, αν υπάρχει", "DOI, if the results have one"),

    # ---- analytics ----------------------------------------------------------------------
    "field.chart_x":      ("Οριζόντιος άξονας", "Chart X axis"),
    "field.chart_y":      ("Κατακόρυφος άξονας", "Chart Y axis"),
    "field.include_kind": ("Ποιες αρχιτεκτονικές", "Which architectures"),

    # ---- space map ----------------------------------------------------------------------
    "field.grid_n":       ("Μεγέθη πληθυσμού n", "Population sizes n"),
    "field.grid_mpcr":    ("Τιμές MPCR", "MPCR values"),
    "field.grid_rounds":  ("Γύροι", "Rounds"),
    "field.grid_seeds":   ("Σπόροι", "Seeds"),
    "field.grid_mix":     ("Συνθέσεις παικτών", "Rosters"),
    "field.cap":          ("Κελιά ανά πάτημα", "Cells to run this click"),

    # ---- evolution ----------------------------------------------------------------------
    "field.population_size":         ("population_size", "population_size"),
    "field.generations":             ("generations", "generations"),
    "field.mutation_rate":           ("mutation_rate", "mutation_rate"),
    "field.mutation_scale":          ("mutation_scale", "mutation_scale"),
    "field.tournament_size":         ("tournament_size", "tournament_size"),
    "field.elite_count":             ("elite_count", "elite_count"),
    "field.matches_per_generation":  ("matches_per_generation", "matches_per_generation"),
    "field.match_rounds":            ("match_rounds", "match_rounds"),
    "field.n_agents_evo":            ("n_agents", "n_agents"),
    "field.pretrain_rounds":         ("pretrain_rounds", "pretrain_rounds"),
    "field.bakeoff":                 ("Τελική αναμέτρηση με τις άλλες αρχιτεκτονικές",
                                      "Final bake-off against the other architectures"),

    # ---- buttons ------------------------------------------------------------------------
    "btn.run":        ("Τρέξε την παρτίδα", "Run the match"),
    "btn.run_sweep":  ("Τρέξε τη σάρωση", "Run the sweep"),
    "btn.run_evolve": ("Τρέξε την εξέλιξη", "Run the evolution"),
    "btn.add":        ("Πρόσθεσε", "Add"),
    "btn.remove":     ("Αφαίρεσε", "Remove"),
    "btn.reset":      ("Καθάρισε", "Reset"),
    "btn.apply":      ("Εφάρμοσε", "Apply"),
    "btn.export":     ("Εξαγωγή", "Export"),
    "btn.import":     ("Εισαγωγή", "Import"),
    "btn.compare":    ("Σύγκρινε τα επιλεγμένα", "Compare selected"),
    "btn.run_missing": ("Τρέξε όσα λείπουν", "Run what is missing"),
}

#: Filled from `T` itself: a label whose two halves are identical is a name, not a translation.
#: Deriving it rather than listing it by hand means the set cannot fall behind the table.
CANONICAL = frozenset(key for key, (el, en) in T.items() if el == en)

#: One sentence per control, in the same `(greek, english)` order. This is what the ⓘ shows.
H: dict[str, tuple[str, str]] = {
    # ---- the game -----------------------------------------------------------------------
    "field.game_kind": (
        "Το σενάριο που θα παίξουν. Κάθε παιχνίδι ορίζει τι επιλογές έχει ο παίκτης, τι βλέπει "
        "πριν αποφασίσει, και πώς προκύπτει η απόδοση του καθενός από τις επιλογές όλων. "
        "Διάλεξε ένα και από κάτω θα εμφανιστεί τι ακριβώς ρωτά. Όσα είναι σημειωμένα ως "
        "published είναι ρυθμισμένα ακριβώς όπως τα έτρεξαν τα αντίστοιχα άρθρα.",
        "The scenario they will play. Each game defines what choices a player has, what it sees "
        "before deciding, and how everyone's payoff follows from everyone's choices. Pick one "
        "and what it actually asks will appear below. Those marked published are configured "
        "exactly as the corresponding papers ran them."),
    "field.rounds": (
        "Πόσες φορές θα παιχτεί το ίδιο παιχνίδι, ο ένας γύρος μετά τον άλλο. Έχει σημασία "
        "γιατί οι περισσότεροι πράκτορες εδώ μαθαίνουν από την εμπειρία: στην αρχή παίζουν "
        "σχεδόν στην τύχη και χρειάζονται χιλιάδες γύρους για να σταθεροποιηθούν. Αν βάλεις "
        "λίγους γύρους θα δεις πώς παίζουν πριν μάθουν, όχι πόσο καλοί είναι. Βάλε 1500 για να "
        "δεις μάθηση, ή 40 για κάτι γρήγορο.",
        "How many times the same game is played, one round after another. It matters because "
        "most agents here learn from experience: at first they play almost at random and need "
        "thousands of rounds to settle. Set it low and you will see how they play before they "
        "learn, not how good they are. Try 1500 to see learning, or 40 for something quick."),
    "field.mpcr": (
        "Πόσο γυρίζει πίσω σε σένα κάθε μονάδα που συνεισφέρεις στο κοινό ταμείο, αφού "
        "πολλαπλασιαστεί και μοιραστεί σε όλους. Στο 0,5 δίνεις μία μονάδα και παίρνεις πίσω "
        "μισή, οπότε όποιος συνεισφέρει μόνος του βγαίνει χαμένος, ενώ αν συνεισφέρουν όλοι "
        "κερδίζουν όλοι. Αυτή ακριβώς η σύγκρουση ανάμεσα στο ατομικό και στο συλλογικό "
        "συμφέρον είναι το δίλημμα. Πάνω από 1 το δίλημμα εξαφανίζεται, γιατί τότε η "
        "συνεισφορά συμφέρει ούτως ή άλλως. Άσ' το κενό για μια έγκυρη προεπιλογή.",
        "How much of each unit you contribute to the pool comes back to you, after it is "
        "multiplied and shared among everyone. At 0.5 you give one unit and get half back, so "
        "contributing alone leaves you worse off, while everyone contributing leaves everyone "
        "better off. That conflict between individual and collective interest is the dilemma. "
        "Above 1 the dilemma disappears, because contributing then pays regardless. Leave it "
        "blank for a valid default."),
    "field.seed": (
        "Ο υπολογιστής χρειάζεται τυχαίους αριθμούς για να παίξει, και αυτός εδώ είναι ο "
        "αριθμός από τον οποίο ξεκινά να τους φτιάχνει. Με τον ίδιο σπόρο παίρνεις ακριβώς την "
        "ίδια παρτίδα όσες φορές κι αν την τρέξεις, που είναι ό,τι χρειάζεσαι για να μπορεί "
        "κάποιος άλλος να επαναλάβει το πείραμά σου. Άλλαξέ τον και τρέξε ξανά για να δεις αν "
        "αυτό που βρήκες ισχύει γενικά ή ήταν σύμπτωση μιας συγκεκριμένης τυχαιότητας.",
        "The computer needs random numbers to play, and this is the number it starts building "
        "them from. With the same seed you get exactly the same match however many times you "
        "run it, which is what lets somebody else repeat your experiment. Change it and run "
        "again to see whether what you found holds generally or was a fluke of one particular "
        "set of random draws."),
    "field.n_agents": (
        "Πόσοι παίζουν ταυτόχρονα στην ίδια παρτίδα. Όσο περισσότεροι, τόσο λιγότερο φαίνεται "
        "τι κάνει ο καθένας, οπότε είναι πιο εύκολο να μην δώσει κανείς και να περιμένει από "
        "τους άλλους. Γι' αυτό η συνεργασία συνήθως καταρρέει πιο γρήγορα σε μεγάλες ομάδες "
        "παρά σε μικρές, και αυτό μπορείς να το δεις εδώ αλλάζοντας μόνο αυτόν τον αριθμό.",
        "How many play at once in the same match. The more there are, the less any individual's "
        "contribution shows, so it becomes easier to give nothing and wait for the others. That "
        "is why cooperation usually collapses faster in big groups than in small ones, and you "
        "can watch it happen here by changing only this number."),
    "field.cong_num_positions": (
        "Πόσα βήματα απέχουν οι παίκτες από το βραβείο. Με 2 αποφασίζουν όλοι ταυτόχρονα μία "
        "φορά και τελείωσε, χωρίς να προλάβει κανείς να δει τι κάνουν οι άλλοι. Με 3 ή "
        "περισσότερα υπάρχει ενδιάμεσος χώρος, οπότε βλέπεις κάποιον να πλησιάζει και "
        "προλαβαίνεις να κάνεις στην άκρη. Αυτή η διαφορά είναι που κάνει το παιχνίδι "
        "ενδιαφέρον: δίνει χρόνο για να συνεννοηθούν σιωπηλά.",
        "How many steps the players are from the prize. At 2 everyone decides once, "
        "simultaneously, and it is over before anyone can see what the others are doing. At 3 "
        "or more there is space in between, so you can watch someone approach and still get out "
        "of the way. That difference is what makes the game interesting: it gives them time to "
        "work it out silently."),
    "field.cong_reward_rule": (
        "Πώς μοιράζεται η ζημιά όταν φτάσουν πολλοί μαζί. Οι κανόνες που διαιρούν με το n "
        "μοιράζουν την ποινή σε όλο τον πληθυσμό, οπότε η ζημιά είναι ίδια είτε συγκρούστηκαν "
        "δύο είτε τέσσερις. Αυτοί που διαιρούν με το k τιμωρούν ανάλογα με το πόσοι πράγματι "
        "έπεσαν πάνω σου, δηλαδή όσο πιο γεμάτος ο δρόμος τόσο χειρότερα. Το δεύτερο μοιάζει "
        "περισσότερο με πραγματική κυκλοφοριακή congestion.",
        "How the damage is shared when several arrive together. The rules that divide by n "
        "spread the penalty across the whole population, so colliding with one other player "
        "costs the same as colliding with three. Those that divide by k punish in proportion to "
        "how many actually ran into you, so the busier the road the worse it gets. The second "
        "behaves more like real traffic."),
    "field.cong_memory_episodes": (
        "Πόσα από τα προηγούμενα γυρίσματα θυμάται ο παίκτης όταν αποφασίζει. Με 0 βλέπει μόνο "
        "πού βρίσκονται όλοι αυτή τη στιγμή, οπότε δεν μπορεί να ξέρει σε ποιον έχει έρθει η "
        "σειρά. Με 1 θυμάται ποιος κέρδισε την προηγούμενη φορά, και μόνο έτσι μπορεί να μάθει "
        "να παίρνουν σειρά. Πρόσεξε ότι κάθε επιπλέον βήμα μνήμης πολλαπλασιάζει τις "
        "καταστάσεις που πρέπει να μάθει, οπότε η εκπαίδευση αργεί πολύ γρήγορα.",
        "How much of what has already happened the player remembers when it decides. At 0 it "
        "sees only where everyone is right now, so it cannot know whose turn it is. At 1 it "
        "remembers who won last time, and only then can it learn to take turns. Note that each "
        "extra step of memory multiplies the situations it has to learn, so training slows down "
        "very quickly."),
    "field.cong_episode_max_rounds": (
        "Πότε να σταματήσει ένα γύρισμα στο οποίο δεν έφτασε κανείς. Χρειάζεται γιατί αν όλοι "
        "αποφασίσουν να περιμένουν, το γύρισμα δεν θα τελείωνε ποτέ και η παρτίδα θα κολλούσε. "
        "Όταν συμβεί, καταγράφεται ως γύρισμα χωρίς νικητή. Δεν είναι σφάλμα: είναι ένα "
        "πραγματικό αποτέλεσμα που σου λέει ότι οι παίκτες έγιναν υπερβολικά διστακτικοί.",
        "When to give up on a round in which nobody arrived. It is needed because if everyone "
        "decides to wait, the round would never end and the match would hang. When it happens "
        "it is recorded as a round with no winner. That is not an error: it is a real result "
        "telling you the players became too cautious."),
    "field.cong_full_reward": (
        "Πόσα παίρνει όποιος φτάσει στο βραβείο μόνος του, χωρίς να συγκρουστεί με κανέναν. "
        "Είναι το μέγιστο που μπορεί να κερδίσει κάποιος σε ένα γύρισμα, και όλα τα άλλα "
        "αποτελέσματα μετριούνται σε σχέση με αυτό. Η τιμή καθαυτή δεν αλλάζει τη συμπεριφορά, "
        "αλλάζει μόνο την κλίμακα των αριθμών που θα δεις.",
        "What a player gets for reaching the prize alone, without colliding with anyone. It is "
        "the most anyone can win in a round, and every other outcome is measured against it. "
        "The value itself does not change behaviour, only the scale of the numbers you see."),
    "field.cong_collapse_at_full": (
        "Αν φτάσουν όλοι ταυτόχρονα, πληρώνονται ακριβώς μηδέν αντί για ένα μικρό ποσό. "
        "Υπάρχει για να μην μπορεί η πλήρης αποτυχία συνεννόησης να μοιάζει με μερική επιτυχία "
        "στα νούμερα. Χωρίς αυτό, μια ομάδα που δεν συνεννοήθηκε ποτέ θα έδειχνε να κερδίζει "
        "κάτι, και η μέτρηση θα σε παραπλανούσε.",
        "If every single player arrives together they are paid exactly zero instead of a small "
        "amount. It exists so that a total failure to coordinate cannot look like partial "
        "success in the numbers. Without it, a group that never coordinated at all would appear "
        "to earn something, and the measurement would mislead you."),

    # ---- what gets computed --------------------------------------------------------------
    "field.compute_nash": (
        "Υπολογίζει τι θα έπαιζαν τέλεια ορθολογικοί παίκτες που σκέφτονται μόνο το συμφέρον "
        "τους. Χρησιμεύει ως σημείο αναφοράς: βλέπεις αν οι πράκτορες που μαθαίνουν κατέληξαν "
        "εκεί που προβλέπει η θεωρία ή κάπου αλλού. Στο Public Goods Game η θεωρία λέει ότι δεν "
        "δίνει κανείς τίποτα, οπότε αν οι πράκτορες συνεργάζονται, αυτό είναι το εύρημα. "
        "Προσθέτει λίγα δευτερόλεπτα.",
        "Works out what perfectly rational players thinking only of themselves would do. It is "
        "a reference point: you can see whether the agents that learned ended up where theory "
        "predicts or somewhere else. In Public Goods, theory says nobody gives anything, so if "
        "the agents cooperate, that is the finding. It adds a few seconds."),
    "field.compute_phi": (
        "Μετρά κατά πόσο το «μυαλό» ενός πράκτορα λειτουργεί ως ένα ενιαίο πράγμα και όχι ως "
        "δύο ανεξάρτητα κομμάτια που τυχαίνει να βρίσκονται μαζί. Ο τρόπος που το ελέγχει "
        "είναι να το κόψει στα δύο και να δει πόση πληροφορία χάνεται. Αφορά μόνο τα μικρά "
        "εξελιγμένα κυκλώματα, και είναι πολύ αργό, οπότε άσ' το κλειστό αν δεν το χρειάζεσαι.",
        "Measures whether an agent's \"mind\" works as one thing rather than as two independent "
        "parts that happen to sit together. The way it checks is to cut it in two and see how "
        "much information is lost. It only applies to the small evolved circuits, and it is "
        "very slow, so leave it off unless you need it."),
    "field.show_metrics": (
        "Διαλέγεις ποιες μετρήσεις θα εμφανιστούν στη σελίδα του αποτελέσματος. Δεν αλλάζει "
        "τίποτα στο πείραμα και δεν επηρεάζει τα νούμερα: υπολογίζονται όλα ούτως ή άλλως. "
        "Απλώς κρατά τη σελίδα διαβάσιμη, γιατί δέκα μετρήσεις ταυτόχρονα είναι δύσκολο να "
        "διαβαστούν. Ξεκίνα με δύο ή τρεις.",
        "You choose which measurements appear on the results page. It changes nothing about the "
        "experiment and does not affect the numbers: they are all computed anyway. It just "
        "keeps the page readable, since ten measures at once is hard to take in. Start with two "
        "or three."),
    "field.log_lines": (
        "Η παρτίδα γράφει μια αφήγηση γύρο-γύρο: ποιος έκανε τι, τι κέρδισε, και όπου μπορεί, "
        "γιατί το έκανε. Εδώ λες πόσες από αυτές τις γραμμές να κρατηθούν. Δεν αλλάζει το "
        "πείραμα, μόνο πόσο αναλυτικό είναι το κείμενο που θα διαβάσεις μετά. Σε μεγάλες "
        "παρτίδες κράτα το μικρό, αλλιώς η σελίδα γεμίζει.",
        "The match writes a round-by-round narration: who did what, what they earned, and where "
        "possible why they did it. Here you say how many of those lines to keep. It does not "
        "change the experiment, only how detailed the transcript you read afterwards is. In "
        "long matches keep it small, or the page fills up."),
    "field.record_repo": (
        "Αποθηκεύει το αποτέλεσμα σε ένα τοπικό αρχείο πειραμάτων, υπογεγραμμένο ψηφιακά ώστε "
        "να φαίνεται αν κάποιος το πείραξε αργότερα. Μόνο έτσι θα εμφανιστεί μετά στις "
        "Αναλύσεις και στη Μετα-ανάλυση, όπου συγκρίνεις πολλά πειράματα μαζί. Αν το αφήσεις "
        "κλειστό, θα δεις το αποτέλεσμα μία φορά και θα χαθεί μόλις φύγεις από τη σελίδα.",
        "Saves the result to a local experiment file, digitally signed so that any later "
        "tampering shows. Only then will it appear in Analytics and Meta-analysis, where you "
        "compare many experiments together. Leave it off and you will see the result once, then "
        "lose it as soon as you leave the page."),
    "field.batch_mode": (
        "Αντί για μία παρτίδα, τρέχει πολλές στη σειρά αλλάζοντας κάθε φορά μία μόνο ρύθμιση "
        "και κρατώντας όλες τις άλλες σταθερές. Έτσι, ό,τι διαφορά δεις στα αποτελέσματα "
        "οφείλεται σίγουρα σε εκείνη τη ρύθμιση και όχι σε κάτι άλλο. Είναι ο βασικός τρόπος "
        "να απαντήσεις «τι κάνει αυτή η παράμετρος;» αντί να μαντεύεις.",
        "Instead of one match it runs several in a row, changing a single setting each time and "
        "holding everything else fixed. That way any difference you see in the results is "
        "definitely down to that setting and not something else. It is the basic way to answer "
        "\"what does this parameter do?\" rather than guessing."),
    "field.batch_param": (
        "Ποια ρύθμιση θα πάρει διαφορετική τιμή σε κάθε παρτίδα. Όλες οι υπόλοιπες μένουν "
        "ακριβώς όπως τις άφησες. Διάλεξε αυτήν για την οποία έχεις ερώτημα: αν σε ενδιαφέρει "
        "αν η συνεργασία εξαρτάται από το μέγεθος της ομάδας, άλλαξε το πλήθος παικτών.",
        "Which setting takes a different value in each match. All the others stay exactly as "
        "you left them. Pick the one you have a question about: if you want to know whether "
        "cooperation depends on group size, vary the number of players."),
    "field.batch_values": (
        "Οι τιμές που θα δοκιμαστούν, χωρισμένες με κόμμα, π.χ. 500, 1000, 2000. Κάθε τιμή "
        "γίνεται μια ξεχωριστή παρτίδα που καταγράφεται χωριστά. Τρεις ή τέσσερις τιμές "
        "αρκούν για να φανεί μια τάση.",
        "The values to try, separated by commas, for example 500, 1000, 2000. Each value "
        "becomes a separate match, recorded separately. Three or four values are enough for a "
        "trend to show."),
    "field.wiz_rounds": (
        "Πόσους γύρους θα παίξει η γρήγορη δοκιμαστική παρτίδα. Είναι το ίδιο πράγμα με το "
        "πεδίο Γύροι της κανονικής φόρμας, απλώς εδώ για να ξεκινήσεις αμέσως.",
        "How many rounds the quick trial match plays. It is the same thing as the Rounds field "
        "on the full form, just here so you can start immediately."),
    "field.wiz_mix": (
        "Μια έτοιμη παρέα παικτών, για να τρέξεις κάτι τώρα χωρίς να διαλέξεις έναν-έναν. "
        "Καλή αφετηρία αν δεν ξέρεις ακόμη ποιον θέλεις να δοκιμάσεις. Μπορείς να την αλλάξεις "
        "μετά όπως θες.",
        "A ready-made set of players, so you can run something now without choosing them one by "
        "one. A good starting point if you do not yet know who you want to test. You can change "
        "it afterwards."),

    # ---- agents ---------------------------------------------------------------------------
    "agent.qlearning": (
        "Κρατά έναν πίνακα με μία γραμμή για κάθε κατάσταση που έχει συναντήσει και μία στήλη "
        "για κάθε κίνηση, και γράφει μέσα πόσο καλά του βγήκε η καθεμία. Μετά διαλέγει το "
        "καλύτερο κελί. Το πλεονέκτημά του είναι ότι μπορείς να δεις τι έμαθε, γιατί ο πίνακας "
        "είναι ορατός. Το μειονέκτημα είναι ότι ο πίνακας μεγαλώνει εκρηκτικά, οπότε δεν "
        "αντέχει μεγάλα παιχνίδια.",
        "It keeps a table with one row for each situation it has met and one column for each "
        "move, and writes in how well each one worked out. Then it picks the best cell. Its "
        "advantage is that you can see what it learned, because the table is right there. Its "
        "drawback is that the table grows explosively, so it cannot handle large games."),
    "agent.dqn": (
        "Ίδια ιδέα με το Q-learning, αλλά αντί για πίνακα χρησιμοποιεί νευρωνικό δίκτυο που "
        "μαντεύει πόσο καλή είναι μια κίνηση. Έτσι αντέχει πολύ μεγαλύτερα παιχνίδια, γιατί το "
        "δίκτυο δεν χρειάζεται να έχει δει την κάθε κατάσταση ξεχωριστά. Το τίμημα είναι ότι "
        "δεν φαίνεται πια καθαρά τι έμαθε, και ότι θέλει περισσότερους γύρους για να "
        "σταθεροποιηθεί.",
        "The same idea as Q-learning, but instead of a table it uses a neural network that "
        "estimates how good a move is. That lets it handle much larger games, because the "
        "network does not need to have seen every situation separately. The price is that what "
        "it learned is no longer clearly visible, and that it needs more rounds to settle."),
    "agent.fep": (
        "Δεν κυνηγά ανταμοιβή όπως οι υπόλοιποι. Κρατά μια εικόνα για το τι πιστεύει ότι θα "
        "κάνουν οι άλλοι, και διαλέγει την κίνηση που θα το εκπλήξει λιγότερο. Όταν πέφτει "
        "έξω, διορθώνει την εικόνα του. Είναι ο μόνος εδώ που μοντελοποιεί ρητά τους άλλους, "
        "οπότε είναι ο πιο κοντινός σε αυτό που λέμε «σκέφτεται τι σκέφτονται οι άλλοι».",
        "It does not chase reward like the others. It keeps a picture of what it believes the "
        "others will do, and picks the move that will surprise it least. When it is wrong, it "
        "corrects the picture. It is the only one here that explicitly models the others, so it "
        "is the closest to what we mean by \"thinking about what others are thinking\"."),
    "agent.markov_brain": (
        "Ένα μικρό κύκλωμα από λογικές πύλες που δεν μαθαίνει καθόλου μέσα στην παρτίδα: "
        "παίζει όπως είναι φτιαγμένο. Βελτιώνεται μόνο μέσω εξέλιξης, δηλαδή φτιάχνοντας "
        "πολλές παραλλαγές, κρατώντας τις καλύτερες και επαναλαμβάνοντας για δεκάδες γενιές. "
        "Αυτό γίνεται στη σελίδα Εξέλιξη, όχι εδώ.",
        "A small circuit of logic gates that does not learn during a match at all: it plays as "
        "it was built. It improves only through evolution, that is, by making many variants, "
        "keeping the best and repeating for dozens of generations. That happens on the "
        "Evolution page, not here."),
    "agent.classic": (
        "Σταθεροί κανόνες που δεν μαθαίνουν ποτέ, όπως «δίνε πάντα» ή «μη δίνεις ποτέ». "
        "Υπάρχουν για να έχεις μέτρο σύγκρισης: αν ένας πράκτορας που μαθαίνει για χιλιάδες "
        "γύρους δεν καταφέρνει να τους ξεπεράσει, τότε κάτι δεν πάει καλά με τη μάθησή του. "
        "Βάλε πάντα τουλάχιστον έναν σε μια σύγκριση.",
        "Fixed rules that never learn, such as always give or never give. They exist to give "
        "you a yardstick: if an agent that has learned for thousands of rounds cannot beat "
        "them, something is wrong with its learning. Always put at least one into a comparison."),
    "agent.llm": (
        "Ρωτά ένα γλωσσικό μοντέλο που τρέχει στον δικό σου υπολογιστή, περιγράφοντάς του την "
        "κατάσταση με λέξεις. Εκείνο απαντά με μια κίνηση και με μια αιτιολογία που μπορείς να "
        "διαβάσεις. Χρειάζεται εγκατεστημένο το Ollama. Πρόσεξε ότι η απάντησή του δεν είναι "
        "ποτέ ακριβώς ίδια δύο φορές, ακόμη και με τις ίδιες ρυθμίσεις, οπότε το πείραμα δεν "
        "επαναλαμβάνεται όπως τα υπόλοιπα.",
        "Asks a language model running on your own machine, describing the situation to it in "
        "words. It answers with a move and with a rationale you can read. It needs Ollama "
        "installed. Note that its answer is never exactly the same twice, even at identical "
        "settings, so the experiment does not repeat the way the others do."),
    "param.alpha": (
        "Πόσο βάρος δίνει στο τελευταίο αποτέλεσμα σε σχέση με όσα ήξερε ήδη. Ψηλά μαθαίνει "
        "γρήγορα αλλά είναι ασταθής, γιατί μια κακή τύχη του αλλάζει γνώμη. Χαμηλά είναι "
        "σταθερός αλλά χρειάζεται πολλούς γύρους. Το 0,1 είναι μια συνηθισμένη αφετηρία.",
        "How much weight it gives the latest outcome against everything it already knew. High "
        "learns fast but is unstable, because one piece of bad luck changes its mind. Low is "
        "steady but needs many rounds. 0.1 is a common starting point."),
    "param.gamma": (
        "Πόσο τον νοιάζει τι θα γίνει αργότερα σε σχέση με το τι κερδίζει τώρα. Κοντά στο 1 "
        "σκέφτεται μακροπρόθεσμα, και αυτό είναι απαραίτητο για να μάθει να συνεργάζεται, "
        "γιατί η συνεργασία αποδίδει μόνο αν κρατήσει. Κοντά στο 0 αρπάζει ό,τι μπορεί αμέσως "
        "και δεν συνεργάζεται ποτέ.",
        "How much it cares about what happens later against what it earns right now. Near 1 it "
        "thinks long term, which is necessary for it to learn to cooperate, because cooperation "
        "only pays if it lasts. Near 0 it grabs whatever it can immediately and never "
        "cooperates."),
    "param.epsilon_decay": (
        "Στην αρχή ο πράκτορας δοκιμάζει τυχαίες κινήσεις για να δει τι γίνεται. Αυτό εδώ "
        "ορίζει πόσο γρήγορα σταματά να πειραματίζεται και αρχίζει να εμπιστεύεται ό,τι έμαθε. "
        "Αν σταματήσει πολύ νωρίς, κλειδώνει στην πρώτη συνήθεια που βρήκε χωρίς να δοκιμάσει "
        "κάτι καλύτερο. Αν αργήσει πολύ, δεν προλαβαίνει να αξιοποιήσει ό,τι έμαθε.",
        "At first the agent tries random moves to see what happens. This sets how quickly it "
        "stops experimenting and starts trusting what it learned. Stop too early and it locks "
        "into the first habit it found without ever trying anything better. Stop too late and "
        "it never gets to use what it learned."),
    "param.epsilon_min": (
        "Πόσο τυχαίος παραμένει για πάντα, ακόμη και στο τέλος. Λίγη μόνιμη τυχαιότητα, γύρω "
        "στο 2%, τον βοηθά να μην κολλήσει σε μια κακή συνήθεια για πάντα. Πολλή τον εμποδίζει "
        "να παίξει καλά, γιατί χαλάει τη στρατηγική του με τυχαίες κινήσεις.",
        "How random it stays for good, even at the end. A little permanent randomness, around "
        "2%, helps it not get stuck in a bad habit forever. Too much stops it playing well, "
        "because random moves keep breaking its strategy."),
    "param.hidden": (
        "Πόσο μεγάλο είναι το νευρωνικό δίκτυο, δηλαδή πόσους εσωτερικούς κόμβους έχει. "
        "Μεγαλύτερο μπορεί να μάθει πιο περίπλοκα μοτίβα, αλλά θέλει περισσότερα παραδείγματα "
        "και κινδυνεύει να «αποστηθίσει» τυχαίες συμπτώσεις αντί να μάθει τον κανόνα. Το 64 "
        "είναι λογικό για παιχνίδια αυτού του μεγέθους.",
        "How big the neural network is, that is, how many internal nodes it has. Bigger can "
        "learn more complex patterns, but needs more examples and risks memorising random "
        "coincidences instead of learning the rule. 64 is reasonable for games this size."),
    "param.lr": (
        "Πόσο μεγάλο βήμα κάνει το δίκτυο κάθε φορά που διορθώνεται. Πολύ μεγάλο και "
        "υπερβάλλει, οπότε η εκπαίδευση ταλαντεύεται χωρίς να συγκλίνει. Πολύ μικρό και "
        "προχωρά τόσο αργά που δεν προλαβαίνει μέσα στους γύρους που του έδωσες. Το 0,001 "
        "είναι η συνηθισμένη αφετηρία.",
        "How big a step the network takes each time it corrects itself. Too big and it "
        "overshoots, so training oscillates without settling. Too small and it moves so slowly "
        "that it never gets there within the rounds you gave it. 0.001 is the usual starting "
        "point."),
    "param.buffer_size": (
        "Το δίκτυο δεν μαθαίνει μόνο από τον τελευταίο γύρο: κρατά παλιές εμπειρίες και τις "
        "ξαναμελετά. Εδώ λες πόσες να κρατά. Μεγάλη μνήμη σταθεροποιεί την εκπαίδευση γιατί "
        "δεν επηρεάζεται από τους τελευταίους λίγους γύρους, αλλά κρατά και πολύ παλιές "
        "καταστάσεις που ίσως δεν ισχύουν πια.",
        "The network does not learn only from the last round: it keeps past experiences and "
        "studies them again. Here you say how many to keep. A large memory stabilises training "
        "because it is not swayed by the last few rounds, but it also holds on to very old "
        "situations that may no longer apply."),
    "param.batch_size": (
        "Πόσες από αυτές τις αποθηκευμένες εμπειρίες κοιτά μαζί κάθε φορά που εκπαιδεύεται. "
        "Περισσότερες δίνουν πιο ομαλή και αξιόπιστη διόρθωση, αλλά κάθε βήμα κοστίζει "
        "περισσότερο χρόνο.",
        "How many of those stored experiences it looks at together each time it trains. More "
        "gives a smoother, more reliable correction, but each step costs more time."),
    "param.train_every": (
        "Κάθε πόσους γύρους σταματά για να εκπαιδευτεί. Με 1 εκπαιδεύεται σε κάθε γύρο, που "
        "σημαίνει γρήγορη μάθηση αλλά αργή παρτίδα. Ανέβασέ το αν η παρτίδα αργεί πολύ και δεν "
        "σε πειράζει να μάθει πιο αργά.",
        "How many rounds pass before it stops to train. At 1 it trains every round, which means "
        "fast learning but a slow match. Raise it if the match is taking too long and you do "
        "not mind slower learning."),
    "param.target_sync_every": (
        "Το δίκτυο συγκρίνει τον εαυτό του με ένα παγωμένο αντίγραφό του, γιατί αλλιώς θα "
        "κυνηγούσε έναν στόχο που κινείται μαζί του και δεν θα σταθεροποιούνταν ποτέ. Εδώ λες "
        "κάθε πόσο ανανεώνεται αυτό το αντίγραφο. Σπάνια χρειάζεται να το πειράξεις.",
        "The network compares itself against a frozen copy of itself, because otherwise it "
        "would be chasing a target that moves along with it and would never settle. Here you "
        "say how often that copy is refreshed. You rarely need to touch it."),
    "param.reciprocity": (
        "Πόσο τον επηρεάζει το τι έκαναν οι άλλοι. Στο 0 κοιτά μόνο το δικό του συμφέρον και "
        "αγνοεί τη συμπεριφορά τους. Όσο το ανεβάζεις, τόσο γίνεται «συνεργάζομαι αν "
        "συνεργάζεσαι»: δίνει όταν πιστεύει ότι δίνουν και οι άλλοι, και σταματά όταν "
        "πιστεύει ότι τον εκμεταλλεύονται.",
        "How much what the others did affects it. At 0 it looks only at its own interest and "
        "ignores their behaviour. As you raise it, it becomes \"I cooperate if you cooperate\": "
        "it gives when it believes the others are giving, and stops when it believes it is "
        "being taken advantage of."),
    "param.obs_noise": (
        "Πόσο αβέβαιος θεωρεί ότι είναι για το τι ακριβώς είδε. Με περισσότερη αβεβαιότητα "
        "εμπιστεύεται λιγότερο την τελευταία παρατήρηση και περισσότερο αυτά που πίστευε ήδη, "
        "οπότε αλλάζει γνώμη πιο δύσκολα. Με λιγότερη, κάθε νέα παρατήρηση τον επηρεάζει πολύ.",
        "How uncertain it considers itself about what exactly it saw. With more uncertainty it "
        "trusts the latest observation less and what it already believed more, so it changes "
        "its mind less readily. With less, every new observation moves it a lot."),
    "param.drift": (
        "Πόσο γρήγορα ξεθωριάζουν όσα πιστεύει, αν δεν επιβεβαιώνονται. Ψηλά ξεχνά γρήγορα, "
        "οπότε προσαρμόζεται καλά σε αντιπάλους που αλλάζουν συμπεριφορά, αλλά χάνει μοτίβα "
        "που εμφανίζονται σπάνια. Χαμηλά κρατά περισσότερα αλλά αργεί να καταλάβει ότι κάτι "
        "άλλαξε.",
        "How quickly what it believes fades if it is not confirmed. High forgets fast, so it "
        "adapts well to opponents that change behaviour, but it loses patterns that appear "
        "rarely. Low keeps more but is slow to notice that something has changed."),
    "param.precision": (
        "Πόσο αποφασιστικά ακολουθεί αυτό που θεωρεί καλύτερο. Ψηλά διαλέγει σχεδόν πάντα την "
        "κορυφαία επιλογή, οπότε είναι προβλέψιμος. Χαμηλά παίζει πιο χαλαρά και πιο τυχαία, "
        "που τον κάνει δυσκολότερο να διαβαστεί αλλά και λιγότερο αποτελεσματικό.",
        "How decisively it follows what it considers best. High almost always takes the top "
        "option, which makes it predictable. Low plays more loosely and more randomly, which "
        "makes it harder to read but also less effective."),
    "param.n_hidden": (
        "Πόσους εσωτερικούς κόμβους έχει το κύκλωμα. Αυτοί είναι η μνήμη του: χωρίς αυτούς δεν "
        "μπορεί να κρατήσει τίποτα από τον έναν γύρο στον επόμενο, οπότε δεν μπορεί να μάθει "
        "να παίρνει σειρά. Περισσότεροι δίνουν περισσότερη μνήμη αλλά κάνουν την εξέλιξη πιο "
        "αργή και τον υπολογισμό του Φ πολύ πιο ακριβό.",
        "How many internal nodes the circuit has. These are its memory: without them it can "
        "keep nothing from one round to the next, so it cannot learn to take turns. More gives "
        "more memory but makes evolution slower and computing Phi far more expensive."),
    "param.strategy": (
        "Ποιον σταθερό κανόνα ακολουθεί. Το AllC δίνει πάντα, το AllD ποτέ, το Random ρίχνει "
        "κέρμα, και το MajorityTFT κάνει ό,τι έκανε η πλειοψηφία τον προηγούμενο γύρο. Το "
        "τελευταίο είναι το πιο ενδιαφέρον, γιατί ανταποδίδει.",
        "Which fixed rule it follows. AllC always gives, AllD never does, Random tosses a coin, "
        "and MajorityTFT does whatever the majority did last round. The last is the most "
        "interesting, because it reciprocates."),
    "param.p": (
        "Με πόση πιθανότητα δίνει, όταν η στρατηγική είναι Random. Το 0,5 είναι δίκαιο κέρμα. "
        "Οι υπόλοιπες στρατηγικές το αγνοούν εντελώς.",
        "How likely it is to give, when the strategy is Random. 0.5 is a fair coin. The other "
        "strategies ignore it entirely."),
    "param.backend_name": (
        "Από πού έρχονται οι απαντήσεις του γλωσσικού μοντέλου. Το ollama ρωτά πραγματικό "
        "μοντέλο στον υπολογιστή σου, οπότε οι απαντήσεις είναι καινούργιες κάθε φορά. Το "
        "replay ξαναπαίζει απαντήσεις που είχαν καταγραφεί σε προηγούμενο τρέξιμο, οπότε "
        "παίρνεις ακριβώς το ίδιο αποτέλεσμα και μπορείς να επαναλάβεις το πείραμα.",
        "Where the language model's answers come from. ollama asks a real model on your "
        "machine, so the answers are fresh every time. replay plays back answers recorded in an "
        "earlier run, so you get exactly the same result and the experiment can be repeated."),
    "param.model": (
        "Ποιο μοντέλο θα ρωτηθεί, με το όνομα που έχει στο Ollama, π.χ. llama3.2:1b. Πρέπει να "
        "το έχεις κατεβάσει πρώτα γράφοντας ollama pull llama3.2:1b στο τερματικό. Τα μικρά "
        "μοντέλα απαντούν σε δευτερόλεπτα αλλά παίζουν φτωχά· τα μεγάλα αργούν πολύ.",
        "Which model to ask, by its name in Ollama, for example llama3.2:1b. You must download "
        "it first by typing ollama pull llama3.2:1b in a terminal. Small models answer in "
        "seconds but play poorly; large ones are much slower."),
    "param.profile": (
        "Τι ακριβώς λέει στο μοντέλο το κείμενο που του στέλνεται. Το minimal-v1 δίνει μόνο "
        "τους κανόνες του παιχνιδιού. Το informed-v1 δίνει επιπλέον τι έκαναν οι άλλοι και "
        "πόσα έχει κερδίσει ως τώρα. Είναι επιλογή από λίστα και όχι ελεύθερο κείμενο, γιατί "
        "αν δύο μοντέλα πάρουν διαφορετικές οδηγίες τότε δεν συγκρίνονται μεταξύ τους.",
        "What exactly the text sent to the model says. minimal-v1 gives only the rules of the "
        "game. informed-v1 additionally gives what the others did and how much it has earned so "
        "far. It is a choice from a list rather than free text, because if two models are given "
        "different instructions they are not being compared."),
    "param.native_schema": (
        "Αναγκάζει το μοντέλο να απαντήσει σε αυστηρά καθορισμένη μορφή που διαβάζει ο "
        "υπολογιστής. Χωρίς αυτό απαντά ελεύθερα, σαν να σου μιλάει, και συχνά χρειάζεται να "
        "ξαναρωτηθεί μέχρι να δώσει κάτι αναγνώσιμο. Άσ' το ανοιχτό εκτός αν θες να μελετήσεις "
        "ακριβώς αυτό.",
        "Forces the model to answer in a strictly defined shape the computer can read. Without "
        "it the model answers freely, as if talking to you, and often has to be asked again "
        "before it produces something readable. Leave it on unless that is exactly what you "
        "want to study."),
    "param.history": (
        "Πόσους προηγούμενους γύρους βλέπει το μοντέλο μέσα στο κείμενο που του στέλνεται. "
        "Περισσότεροι γύροι σημαίνουν καλύτερη μνήμη, αλλά και μεγαλύτερο κείμενο, άρα πιο "
        "αργή και πιο ακριβή απάντηση.",
        "How many previous rounds the model sees inside the text sent to it. More rounds means "
        "better memory, but also a longer text, so a slower and more expensive answer."),
    "param.persona": (
        "Προαιρετική περιγραφή ρόλου, π.χ. «είσαι προσεκτικός διαπραγματευτής». Αλλάζει τη "
        "συμπεριφορά χωρίς να αλλάζει το παιχνίδι, οπότε άσ' το κενό αν θέλεις καθαρή σύγκριση "
        "ανάμεσα σε μοντέλα.",
        "An optional description of a role, for example \"you are a cautious negotiator\". It "
        "changes behaviour without changing the game, so leave it empty if you want a clean "
        "comparison between models."),
    "param.p_cooperate": (
        "Με πόση πιθανότητα δίνει, όταν η στρατηγική είναι Random. Το 0,5 είναι δίκαιο κέρμα. "
        "Οι υπόλοιπες στρατηγικές το αγνοούν.",
        "How likely it is to give, when the strategy is Random. 0.5 is a fair coin. The other "
        "strategies ignore it."),
    "param.llm_model": (
        "Ποιο μοντέλο θα ρωτηθεί, με το όνομα που έχει στο Ollama, π.χ. llama3.2:1b. Πρέπει να "
        "το έχεις κατεβάσει πρώτα με ollama pull.",
        "Which model to ask, by its name in Ollama, for example llama3.2:1b. You must have "
        "pulled it first with ollama pull."),
    "param.llm_constrained": (
        "Με 1 ο διακομιστής αναγκάζει το μοντέλο να απαντήσει σε αυστηρή μορφή. Με 0 του το "
        "ζητά μόνο το κείμενο, οπότε συχνά χρειάζεται δεύτερη προσπάθεια για να δώσει κάτι "
        "αναγνώσιμο.",
        "At 1 the server forces the model to answer in a strict shape. At 0 only the text asks "
        "for it, so a second attempt is often needed before it produces something readable."),
    "param.llm_profile": (
        "Τι ακριβώς λέει στο μοντέλο το κείμενο που του στέλνεται. Το minimal-v1 δίνει μόνο "
        "τους κανόνες, το informed-v1 δίνει και το τι έκαναν οι άλλοι. Επιλογή από λίστα και "
        "όχι ελεύθερο κείμενο, γιατί δύο μοντέλα με διαφορετικές οδηγίες δεν συγκρίνονται.",
        "What exactly the text sent to the model says. minimal-v1 gives only the rules, "
        "informed-v1 also gives what the others did. It is a choice from a list rather than "
        "free text, because two models given different instructions are not being compared."),
    "field.genome_origin": (
        "Το κύκλωμα δεν μαθαίνει μέσα στην παρτίδα: παίζει όπως είναι φτιαγμένο. Ένα καινούργιο "
        "τυχαίο θα παίξει σχεδόν στην τύχη, που είναι χρήσιμο μόνο ως σημείο αναφοράς. Για να "
        "δεις τι μπορεί να κάνει, πρέπει πρώτα να το εξελίξεις στη σελίδα Εξέλιξη και μετά να "
        "φέρεις εδώ τον κωδικό του. Η εξέλιξη είναι προετοιμασία πριν την παρτίδα, δεν τρέχει "
        "μέσα σε αυτήν.",
        "The circuit does not learn during a match: it plays as it was built. A fresh random one "
        "plays close to randomly, which is useful only as a baseline. To see what it can do you "
        "first evolve it on the Evolution page and then bring its identifier here. Evolution is "
        "preparation before a match, not something that runs inside one."),
    "param.genome_cid": (
        "Προαιρετικό. Όταν τρέξεις εξέλιξη, το καλύτερο κύκλωμα που θα προκύψει παίρνει έναν "
        "μοναδικό κωδικό. Αν τον αντιγράψεις εδώ, θα παίξει εκείνο αντί για ένα καινούργιο "
        "τυχαίο. Έτσι δοκιμάζεις αν αυτό που εξελίχθηκε στέκεται και απέναντι σε άλλους "
        "αντιπάλους.",
        "Optional. When you run evolution, the best circuit it produces gets a unique "
        "identifier. Paste it here and that circuit plays instead of a fresh random one. That "
        "is how you test whether what evolved also holds up against different opponents."),
    "param.epsilon_pair": (
        "Οι δύο αριθμοί μαζί ελέγχουν τον πειραματισμό. Ο πρώτος λέει πόσο γρήγορα σταματά να "
        "δοκιμάζει τυχαίες κινήσεις, ο δεύτερος πόσο τυχαίος μένει για πάντα. Η γραμμή από "
        "κάτω υπολογίζει πόσους γύρους χρειάζεται για να φτάσει εκεί, και σου λέει αν οι γύροι "
        "που έβαλες αρκούν. Αν δεν αρκούν, ο πράκτορας θα κρίνεται ενώ ακόμη πειραματίζεται.",
        "The two numbers together control experimentation. The first says how quickly it stops "
        "trying random moves, the second how random it stays for good. The line below works out "
        "how many rounds it needs to get there and tells you whether the rounds you set are "
        "enough. If they are not, the agent is being judged while it is still experimenting."),
    "field.row_kind": (
        "Ποια αρχιτεκτονική θα καθίσει σε αυτές τις θέσεις. Μπορείς να βάλεις πολλές "
        "διαφορετικές στην ίδια παρτίδα, και εκεί ακριβώς είναι το ενδιαφέρον: να δεις πώς τα "
        "πάει ένας τύπος μυαλού απέναντι σε έναν άλλο.",
        "Which architecture sits in these seats. You can put several different ones into the "
        "same match, and that is exactly where it gets interesting: seeing how one kind of mind "
        "does against another."),
    "field.row_count": (
        "Πόσες θέσεις παίρνει αυτή η αρχιτεκτονική. Το συνολικό πλήθος παικτών βγαίνει "
        "προσθέτοντας όλες τις σειρές, δεν το ορίζεις χωριστά. Βάλε 0 για να τη βγάλεις "
        "προσωρινά χωρίς να χάσεις τις ρυθμίσεις της.",
        "How many seats this architecture takes. The total number of players comes from adding "
        "up all the rows; you do not set it separately. Put 0 to leave it out temporarily "
        "without losing its settings."),

    # ---- measures --------------------------------------------------------------------------
    "metric.cooperation_rate": (
        "Το ποσοστό των γύρων στους οποίους ο παίκτης έδωσε στο κοινό ταμείο, ή έκανε στην "
        "άκρη στα παιχνίδια congestion. Είναι η πιο άμεση μέτρηση του αν συνεργάστηκε: 1 σημαίνει πάντα, "
        "0 ποτέ.",
        "The share of rounds in which the player gave to the pot, or gave way in Congestion. It "
        "is the most direct measure of whether it cooperated: 1 means always, 0 never."),
    "metric.efficiency": (
        "Πόσο κοντά έφτασε η ομάδα στο καλύτερο δυνατό συνολικό αποτέλεσμα. Το 1 σημαίνει ότι "
        "δεν πήγε τίποτα χαμένο, το 0 ότι πήγαν όλα χαμένα. Πρόσεξε ότι μια ομάδα μπορεί να "
        "είναι αποδοτική και άδικη ταυτόχρονα, αν τα πήρε όλα ένας.",
        "How close the group came to the best total outcome available. 1 means nothing was "
        "wasted, 0 that everything was. Note that a group can be efficient and unfair at the "
        "same time, if one player took it all."),
    "metric.payoff_gini": (
        "Πόσο άνισα μοιράστηκαν τα κέρδη. Το 0 σημαίνει ότι όλοι πήραν ακριβώς τα ίδια, το 1 "
        "ότι τα πήρε όλα ένας. Είναι ο ίδιος δείκτης που χρησιμοποιείται για την εισοδηματική "
        "ανισότητα στις χώρες.",
        "How unequally the payoffs were shared. 0 means everyone got exactly the same, 1 that "
        "one player took everything. It is the same index used for income inequality between "
        "countries."),
    "metric.action_entropy_bits": (
        "Πόσο απρόβλεπτες ήταν οι κινήσεις του. Κοντά στο 0 έκανε σχεδόν πάντα το ίδιο πράγμα, "
        "οπότε ήταν εύκολο να τον διαβάσει κανείς. Το 1 bit σημαίνει τέλεια απρόβλεπτος "
        "ανάμεσα σε δύο επιλογές, δηλαδή σαν κέρμα.",
        "How unpredictable its moves were. Near 0 it did almost the same thing every time, so "
        "it was easy to read. 1 bit means perfectly unpredictable between two options, that is, "
        "like a coin."),
    "metric.mutual_information_bits": (
        "Πόσο η κίνηση του παίκτη εξαρτήθηκε πραγματικά από αυτό που είδε. Κοντά στο 0 "
        "σημαίνει ότι αγνοούσε την κατάσταση και έπαιζε από συνήθεια, ακόμη κι αν τα πήγαινε "
        "καλά. Είναι χρήσιμο γιατί ξεχωρίζει το «έμαθε να αντιδρά» από το «βρήκε μια κίνηση "
        "που δουλεύει».",
        "How far the player's move actually depended on what it saw. Near 0 means it was "
        "ignoring the situation and playing out of habit, even if it was doing well. It is "
        "useful because it separates \"learned to react\" from \"found one move that works\"."),
    "metric.transfer_entropy_bits": (
        "Πόσο οι κινήσεις ενός παίκτη σε βοηθούν να προβλέψεις τις επόμενες κινήσεις ενός "
        "άλλου, πέρα από όσα ήξερες ήδη. Είναι μέτρηση επιρροής με κατεύθυνση: δείχνει ποιος "
        "αντιδρά σε ποιον. Κρατιούνται μόνο τα ζεύγη που περνούν στατιστικό έλεγχο, ώστε να "
        "μην αναφέρονται συμπτώσεις ως ευρήματα.",
        "How much one player's moves help you predict another's next moves, beyond what you "
        "already knew. It is a directed measure of influence: it shows who is reacting to whom. "
        "Only pairs that pass a statistical test are kept, so coincidences are not reported as "
        "findings."),
    "metric.predictive_information_bits": (
        "Πόσο το παρελθόν ενός παίκτη προβλέπει το μέλλον του. Ψηλές τιμές σημαίνουν ότι "
        "σχημάτισε σταθερό μοτίβο, π.χ. εναλλάσσεται τακτικά. Κοντά στο 0 σημαίνει ότι κάθε "
        "γύρος του είναι ανεξάρτητος από τους προηγούμενους.",
        "How much a player's past predicts its own future. High values mean it formed a stable "
        "pattern, for example alternating regularly. Near 0 means each of its rounds is "
        "independent of the previous ones."),
    "metric.coop_graph_weight": (
        "Φτιάχνεται ένα δίκτυο όπου δύο παίκτες συνδέονται όσο πιο συχνά έδωσαν ταυτόχρονα. "
        "Αυτός ο αριθμός λέει πόσο πυκνό είναι το δίκτυο συνολικά, δηλαδή πόσο γενικευμένη "
        "ήταν η αμοιβαία συνεργασία.",
        "A network is built in which two players are linked the more often they gave at the "
        "same time. This number says how dense that network is overall, that is, how widespread "
        "mutual cooperation was."),
    "metric.coop_graph_clustering": (
        "Στο ίδιο δίκτυο, δείχνει αν η συνεργασία έγινε σε κλειστές παρέες, όπου οι ίδιοι "
        "λίγοι συνεργάζονται μεταξύ τους, ή αν απλώθηκε ομοιόμορφα σε όλους. Ψηλή τιμή "
        "σημαίνει κλίκες.",
        "In the same network, it shows whether cooperation happened in tight circles, where the "
        "same few cooperate with each other, or spread evenly across everyone. A high value "
        "means cliques."),
    "metric.influence_graph_density": (
        "Πόσα ζεύγη παικτών επηρέασαν μετρήσιμα ο ένας τον άλλον. Ψηλή τιμή σημαίνει ότι όλοι "
        "παρακολουθούσαν και αντιδρούσαν σε όλους· χαμηλή ότι ο καθένας έπαιζε ουσιαστικά "
        "μόνος του.",
        "How many pairs of players measurably influenced one another. A high value means "
        "everyone was watching and reacting to everyone; a low one that each was essentially "
        "playing alone."),
    "metric.phi": (
        "Πόσο το κύκλωμα ενός πράκτορα λειτουργεί ως ένα ενιαίο πράγμα και όχι ως ξεχωριστά "
        "κομμάτια. Ο υπολογισμός το κόβει στα δύο με κάθε δυνατό τρόπο και μετρά πόση "
        "πληροφορία χάνεται στο ασθενέστερο κόψιμο. Έχει νόημα μόνο για τα μικρά εξελιγμένα "
        "κυκλώματα και είναι αργό.",
        "How far an agent's circuit works as one thing rather than as separate parts. The "
        "computation cuts it in two in every possible way and measures how much information is "
        "lost at the weakest cut. It only means something for the small evolved circuits, and "
        "it is slow."),

    # ---- sweep ------------------------------------------------------------------------------
    "field.axis": (
        "Ποια ρύθμιση θα αλλάζει από τρέξιμο σε τρέξιμο. Όλες οι υπόλοιπες μένουν σταθερές, "
        "οπότε ό,τι κινηθεί στα αποτελέσματα οφείλεται σε αυτήν. Είναι ο καθαρός τρόπος να "
        "απαντήσεις «τι κάνει αυτή η παράμετρος;».",
        "Which setting changes from run to run. All the others stay fixed, so whatever moves in "
        "the results is down to this one. It is the clean way to answer \"what does this "
        "parameter do?\"."),
    "field.values": (
        "Οι τιμές που θα δοκιμαστούν, χωρισμένες με κόμμα, π.χ. 0.3, 0.4, 0.5, 0.6. Κάθε τιμή "
        "γίνεται ξεχωριστό πείραμα. Βάλε τέσσερις ή πέντε για να φανεί σχήμα και όχι μόνο δύο "
        "σημεία.",
        "The values to try, separated by commas, for example 0.3, 0.4, 0.5, 0.6. Each becomes "
        "its own experiment. Use four or five so a shape appears rather than just two points."),
    "field.seeds": (
        "Πόσες φορές θα τρέξει κάθε τιμή με διαφορετική τυχαιότητα. Μία φορά δεν λέει τίποτα, "
        "γιατί δεν μπορείς να ξεχωρίσεις το πραγματικό αποτέλεσμα από την τύχη μιας παρτίδας. "
        "Με πέντε αρχίζεις να βλέπεις αν η διαφορά που παρατήρησες είναι σταθερή.",
        "How many times each value runs with different randomness. Once tells you nothing, "
        "because you cannot separate a real effect from the luck of one match. At five you "
        "start to see whether the difference you noticed is consistent."),
    "field.workers": (
        "Πόσα τρεξίματα γίνονται ταυτόχρονα, μοιρασμένα στους πυρήνες του υπολογιστή σου. "
        "Περισσότερα τελειώνουν γρηγορότερα αλλά ζορίζουν το μηχάνημα. Τα αποτελέσματα είναι "
        "ακριβώς τα ίδια όπως κι αν τα μοιράσεις, γιατί κάθε τρέξιμο είναι ανεξάρτητο.",
        "How many runs happen at once, spread across your machine's cores. More finishes sooner "
        "but loads the machine. The results are exactly the same however you split them, "
        "because each run is independent."),
    "field.mix": (
        "Ποια παρέα παικτών θα παίζει σε κάθε τρέξιμο. Μένει η ίδια σε όλες τις τιμές, αλλιώς "
        "δεν θα ήξερες αν η διαφορά ήρθε από την παράμετρο ή από το ότι άλλαξαν οι παίκτες.",
        "Which set of players runs at every value. It stays the same throughout, otherwise you "
        "would not know whether a difference came from the parameter or from the players "
        "changing."),

    # ---- coverage ---------------------------------------------------------------------------
    "field.unit": (
        "Τι μετράει ένα σημείο στον χάρτη. Αν διαλέξεις «πείραμα», τα τρεξίματα που είναι "
        "ουσιαστικά το ίδιο πείραμα με άλλον σπόρο μετρούν ως ένα, οπότε βλέπεις πόσες "
        "διαφορετικές ερωτήσεις έχεις κάνει. Αν διαλέξεις «τρέξιμο», μετρά το καθένα χωριστά, "
        "οπότε βλέπεις πόση δουλειά έχει γίνει.",
        "What one point on the map counts. Choose design and runs that are essentially the same "
        "experiment with a different seed count as one, so you see how many distinct questions "
        "you have asked. Choose record and each is counted separately, so you see how much work "
        "has been done."),
    "field.family": (
        "Ποια πεδία μπαίνουν στον πίνακα συσχετίσεων. Αν διαλέξεις μόνο τα μέτρα, βλέπεις πώς "
        "σχετίζονται τα αποτελέσματα μεταξύ τους. Αν βάλεις τα πάντα, μπαίνουν και οι "
        "ρυθμίσεις, που συχνά συσχετίζονται τετριμμένα μεταξύ τους και γεμίζουν τον πίνακα με "
        "θόρυβο.",
        "Which fields enter the correlation table. Choose measures only and you see how the "
        "results relate to each other. Choose everything and the settings come in too, which "
        "often correlate with each other trivially and fill the table with noise."),
    "field.limit": (
        "Πόσα πεδία το πολύ θα μπουν στον πίνακα. Πάνω από περίπου είκοσι, ο πίνακας γίνεται "
        "τόσο μεγάλος που δεν διαβάζεται. Κράτα τον μικρό και εστίασε σε ό,τι σε ενδιαφέρει.",
        "How many fields at most enter the table. Beyond about twenty it becomes so large that "
        "it cannot be read. Keep it small and focus on what interests you."),

    # ---- meta-analysis ----------------------------------------------------------------------
    "field.metric": (
        "Ποιο μέτρο θα συνδυαστεί ανάμεσα στα πειράματα. Γίνεται ένα μέτρο τη φορά, γιατί "
        "διαφορετικά μέτρα έχουν διαφορετικές κλίμακες και δεν έχει νόημα να μπουν στον ίδιο "
        "μέσο όρο.",
        "Which measure is combined across experiments. It is done one measure at a time, "
        "because different measures live on different scales and averaging them together would "
        "mean nothing."),
    "field.only_local": (
        "Αν είναι επιλεγμένο, αγνοεί τα αποτελέσματα που εισήγαγες από άλλες εγκαταστάσεις και "
        "δείχνει μόνο τα δικά σου. Χρήσιμο για να δεις τι λένε μόνο τα δικά σου δεδομένα πριν "
        "τα συγκρίνεις με ξένα.",
        "When ticked, ignores results you imported from other installations and shows only your "
        "own. Useful for seeing what your own data alone says before comparing it with anyone "
        "else's."),
    "field.bundle": (
        "Ένα αρχείο zip που έφτιαξε άλλη εγκατάσταση με τα αποτελέσματά της. Θα ελεγχθεί με το "
        "κλειδί που θα δώσεις παρακάτω, και αν δεν επαληθεύεται θα απορριφθεί ολόκληρο. Τα "
        "ξένα αποτελέσματα κρατιούνται χωριστά και δεν ανακατεύονται με τα δικά σου.",
        "A zip another installation made with its results. It will be checked against the key "
        "you give below, and if it does not verify the whole thing is refused. Other people's "
        "results are kept separately and never mixed into your own."),
    "field.trusted_key": (
        "Το δημόσιο κλειδί εκείνου που έφτιαξε το αρχείο, όπως στο έδωσε. Πρέπει να το πάρεις "
        "από αλλού, π.χ. με email, και όχι μέσα από το ίδιο το αρχείο: ένα αρχείο που "
        "εγγυάται για τον εαυτό του δεν αποδεικνύει τίποτα. Χωρίς κλειδί η εισαγωγή δεν "
        "γίνεται.",
        "The public key of whoever made the bundle, as they gave it to you. You must get it "
        "some other way, by email say, and not from inside the bundle itself: a file that "
        "vouches for itself proves nothing. Without a key the import does not happen."),
    "field.label": (
        "Ένα σύντομο όνομα για να ξεχωρίζεις από πού ήρθαν αυτά τα αποτελέσματα, π.χ. "
        "«Εργαστήριο Α» ή «Μαρία». Εμφανίζεται δίπλα στα νούμερά τους σε κάθε σύγκριση.",
        "A short name so you can tell where these results came from, for example \"Lab A\" or a "
        "colleague's name. It appears beside their numbers in every comparison."),
    "field.title": (
        "Τίτλος για το αρχείο που θα στείλεις, ώστε να καταλάβει ο παραλήπτης τι περιέχει, "
        "π.χ. «Τρεξίματα Σεπτεμβρίου, δημόσια αγαθά».",
        "A title for the bundle you send, so whoever receives it understands what is in it, for "
        "example \"September runs, public goods\"."),
    "field.doi": (
        "Αν τα αποτελέσματα έχουν ήδη δημοσιευτεί και έχουν DOI, γράψ' το εδώ ώστε να ταξιδεύει "
        "μαζί τους και να μπορεί κάποιος να βρει τη δημοσίευση από το αρχείο.",
        "If the results are already published and have a DOI, put it here so it travels with "
        "them and somebody can find the publication from the bundle."),

    # ---- analytics --------------------------------------------------------------------------
    "field.chart_x": (
        "Τι μπαίνει στον οριζόντιο άξονα. Συνήθως βάζεις εδώ μια ρύθμιση που άλλαξες, π.χ. το "
        "πλήθος παικτών, για να δεις πώς επηρεάζει το αποτέλεσμα.",
        "What goes on the horizontal axis. You usually put a setting you varied here, the "
        "number of players say, to see how it affects the outcome."),
    "field.chart_y": (
        "Τι μπαίνει στον κατακόρυφο άξονα. Συνήθως ένα μέτρο αποτελέσματος, π.χ. ο ρυθμός "
        "συνεργασίας.",
        "What goes on the vertical axis. Usually an outcome measure, the cooperation rate say."),
    "field.include_kind": (
        "Ποιες αρχιτεκτονικές θα μπουν στη σύγκριση. Ξεδιάλεξε όσες δεν σε ενδιαφέρουν για να "
        "μείνουν μόνο αυτές που θες να συγκρίνεις.",
        "Which architectures enter the comparison. Untick the ones you are not interested in so "
        "only those you want to compare remain."),
    "field.filter_field": (
        "Ποιο αποθηκευμένο στοιχείο θα φιλτράρει. Μπορεί να είναι ρύθμιση του παιχνιδιού, "
        "μέτρο αποτελέσματος, ή παράμετρος μιας συγκεκριμένης αρχιτεκτονικής.",
        "Which stored item to filter on. It can be a game setting, an outcome measure, or a "
        "parameter of one particular architecture."),
    "field.filter_operator": (
        "Πώς συγκρίνεται το στοιχείο με την τιμή που θα δώσεις: ίσο, μεγαλύτερο, μικρότερο και "
        "λοιπά.",
        "How the item is compared against the value you give: equal to, greater than, less "
        "than and so on."),
    "field.filter_value": (
        "Με τι συγκρίνεται. Αν φιλτράρεις παράμετρο μιας αρχιτεκτονικής, το τρέξιμο περνά "
        "εφόσον την ικανοποιεί έστω ένας πράκτορας εκείνου του είδους μέσα στην παρτίδα.",
        "What it is compared against. If you filter on an architecture's parameter, a run "
        "passes as long as at least one agent of that kind in the match satisfies it."),
    "field.group_by": (
        "Μάζεψε τα τρεξίματα σε γραμμές αντί να τα δείχνεις ένα-ένα. Η ομαδοποίηση ανά πείραμα "
        "είναι η πιο χρήσιμη: βάζει μαζί όσα ρωτούν την ίδια ερώτηση με διαφορετική τυχαιότητα, "
        "οπότε βλέπεις πόσες διαφορετικές ερωτήσεις έχεις κάνει και όχι πόσες φορές πάτησες "
        "τρέξιμο. Το γράφημα και η συσχέτιση πάνω δεν αλλάζουν: βλέπουν πάντα όλα τα "
        "τρεξίματα.",
        "Collapse runs into rows instead of listing them one by one. Grouping by design is the "
        "most useful: it puts together the runs that ask the same question with different "
        "randomness, so you see how many distinct questions you have asked rather than how many "
        "times you pressed run. The chart and the correlation above do not change: they always "
        "see every run."),
    "field.per_page": (
        "Πόσες γραμμές ανά σελίδα. Επηρεάζει μόνο τον πίνακα, όχι τι μπαίνει στο γράφημα ή "
        "στους υπολογισμούς.",
        "How many rows per page. It affects only the table, not what enters the chart or the "
        "calculations."),
    "field.roster_has": (
        "Κράτα μόνο τα τρεξίματα στα οποία συμμετείχε τουλάχιστον ένας πράκτορας από τα είδη "
        "που θα τσεκάρεις.",
        "Keep only runs in which at least one agent of the kinds you tick took part."),

    # ---- space map --------------------------------------------------------------------------
    "field.grid_n": (
        "Τα μεγέθη ομάδας που θέλεις να καλυφθούν, χωρισμένα με κόμμα, π.χ. 2, 3, 5. Ο χάρτης "
        "θα σου δείξει ποιοι συνδυασμοί έχουν ήδη τρέξει και ποιοι λείπουν.",
        "The group sizes you want covered, separated by commas, for example 2, 3, 5. The map "
        "will show you which combinations have already run and which are missing."),
    "field.grid_mpcr": (
        "Οι τιμές MPCR που θέλεις να καλυφθούν. Θυμήσου ότι το δίλημμα υπάρχει μόνο κάτω από "
        "1, και ότι πολύ μικρές τιμές δεν είναι έγκυρες σε μικρές ομάδες. Τα άκυρα κελιά "
        "εμφανίζονται ως τέτοια αντί να κρύβονται.",
        "The MPCR values you want covered. Remember the dilemma exists only below 1, and that "
        "very small values are not valid in small groups. Invalid cells are shown as such "
        "rather than hidden."),
    "field.grid_rounds": (
        "Τα μήκη παρτίδας που θέλεις να καλυφθούν. Χρήσιμο για να δεις αν ένα εύρημα κρατά και "
        "σε σύντομες και σε μακριές παρτίδες.",
        "The match lengths you want covered. Useful for seeing whether a finding holds in both "
        "short and long matches."),
    "field.grid_seeds": (
        "Πόσες επαναλήψεις με διαφορετική τυχαιότητα ανά κελί. Το ίδιο σκεπτικό με τη σάρωση: "
        "μία επανάληψη δεν ξεχωρίζει το αποτέλεσμα από την τύχη.",
        "How many repeats with different randomness per cell. The same reasoning as the sweep: "
        "one repeat cannot separate a result from luck."),
    "field.grid_mix": (
        "Ποιες παρέες παικτών μπαίνουν στον χάρτη. Πρόσεξε ότι κάθε επιλογή πολλαπλασιάζει τα "
        "κελιά που πρέπει να τρέξουν, οπότε δύο ή τρεις αρκούν για αρχή.",
        "Which sets of players enter the map. Note that each one you tick multiplies the cells "
        "that have to run, so two or three are enough to start."),
    "field.cap": (
        "Πόσα κελιά θα τρέξουν με αυτό το πάτημα. Τρέχουν το ένα μετά το άλλο και η σελίδα "
        "περιμένει μέχρι να τελειώσουν, οπότε κράτα το μικρό αν ο χάρτης περιλαμβάνει DQN, που "
        "είναι το πιο αργό.",
        "How many cells run on this click. They run one after another while the page waits, so "
        "keep it small if the map includes DQN, which is the slowest."),

    # ---- evolution --------------------------------------------------------------------------
    "field.population_size": (
        "Πόσα διαφορετικά κυκλώματα υπάρχουν ταυτόχρονα σε κάθε γενιά. Μεγαλύτερος πληθυσμός "
        "ψάχνει περισσότερες πιθανές λύσεις, άρα βρίσκει καλύτερα αποτελέσματα, αλλά κάθε "
        "γενιά αργεί ανάλογα.",
        "How many different circuits exist at once in each generation. A larger population "
        "searches more possible solutions, so it finds better results, but each generation "
        "takes proportionally longer."),
    "field.generations": (
        "Πόσες φορές θα επαναληφθεί ο κύκλος «παίξε, κράτα τους καλύτερους, φτιάξε απογόνους». "
        "Εδώ γίνεται όλη η μάθηση, αφού το κύκλωμα δεν μαθαίνει μέσα στην παρτίδα. Λίγες "
        "γενιές σημαίνει ότι θα δεις σχεδόν τυχαία κυκλώματα.",
        "How many times the cycle of play, keep the best, make offspring is repeated. All the "
        "learning happens here, since the circuit does not learn inside a match. Too few "
        "generations and you will be looking at near-random circuits."),
    "field.mutation_rate": (
        "Πόσο συχνά αλλάζει τυχαία κάτι στο κύκλωμα ενός απογόνου. Χωρίς μεταλλάξεις δεν "
        "εμφανίζεται ποτέ τίποτα καινούργιο και ο πληθυσμός παγώνει. Με πάρα πολλές, ό,τι καλό "
        "βρέθηκε καταστρέφεται στην επόμενη γενιά.",
        "How often something in an offspring's circuit changes at random. With no mutation "
        "nothing new ever appears and the population freezes. With too much, whatever good was "
        "found is destroyed in the next generation."),
    "field.mutation_scale": (
        "Πόσο μεγάλη είναι κάθε αλλαγή όταν συμβαίνει. Μικρές αλλαγές βελτιώνουν σταδιακά ό,τι "
        "υπάρχει ήδη. Μεγάλες πετάνε το κύκλωμα σε τελείως διαφορετική συμπεριφορά, που μπορεί "
        "να βοηθήσει αν έχει κολλήσει.",
        "How large each change is when it happens. Small changes gradually refine what is "
        "already there. Large ones throw the circuit into entirely different behaviour, which "
        "can help if it is stuck."),
    "field.tournament_size": (
        "Για να διαλεγεί ένας γονέας, μαζεύονται τόσοι τυχαίοι υποψήφιοι και κερδίζει ο "
        "καλύτερος. Μεγαλύτερος αριθμός ευνοεί πιο έντονα τους ήδη καλούς, που επιταχύνει αλλά "
        "μειώνει την ποικιλία και μπορεί να κολλήσει σε τοπική λύση.",
        "To pick a parent, this many random candidates are gathered and the best one wins. A "
        "bigger number favours the already-good more strongly, which speeds things up but "
        "reduces variety and can get stuck on a local solution."),
    "field.elite_count": (
        "Πόσοι από τους καλύτερους περνούν στην επόμενη γενιά χωρίς καμία αλλαγή. Εγγυάται ότι "
        "ό,τι καλό βρέθηκε δεν θα χαθεί από κακοτυχία στις μεταλλάξεις. Δύο ή τρεις αρκούν.",
        "How many of the best pass to the next generation unchanged. It guarantees that what "
        "was found is not lost to bad luck in the mutations. Two or three is enough."),
    "field.matches_per_generation": (
        "Πόσες παρτίδες παίζει κάθε κύκλωμα πριν κριθεί. Με λίγες, κρίνεται από την τύχη μιας "
        "παρτίδας και όχι από την ικανότητά του, οπότε η εξέλιξη κυνηγά θόρυβο.",
        "How many matches each circuit plays before being judged. With too few it is judged on "
        "the luck of one match rather than on ability, so evolution ends up chasing noise."),
    "field.match_rounds": (
        "Πόσοι γύροι έχει κάθε παρτίδα αξιολόγησης. Αρκετοί ώστε να φανεί η συμπεριφορά, αλλά "
        "όχι τόσοι που να αργεί κάθε γενιά.",
        "How many rounds each evaluation match has. Enough for the behaviour to show, but not "
        "so many that every generation drags."),
    "field.n_agents_evo": (
        "Πόσα κυκλώματα παίζουν μαζί σε κάθε παρτίδα αξιολόγησης. Πρέπει να διαιρεί ακριβώς το "
        "μέγεθος του πληθυσμού, ώστε να μπορούν να χωριστούν σε ομάδες.",
        "How many circuits play together in each evaluation match. It has to divide the "
        "population size exactly, so they can be split into groups."),
    "field.pretrain_rounds": (
        "Πόσους γύρους εκπαιδεύονται πρώτα, χωριστά, οι αντίπαλοι που μαθαίνουν. Χωρίς αυτό η "
        "σύγκριση θα ήταν άδικη: το εξελιγμένο κύκλωμα έχει περάσει δεκάδες γενιές, ενώ ο "
        "αντίπαλος θα ξεκινούσε από το μηδέν και θα έχανε επειδή δεν πρόλαβε να μάθει.",
        "How many rounds the opponents that learn are trained first, on their own. Without it "
        "the comparison would be unfair: the evolved circuit has been through dozens of "
        "generations, while the opponent would start from nothing and lose only because it "
        "never got to learn."),
    "field.bakeoff": (
        "Στο τέλος της εξέλιξης, παίρνει το καλύτερο κύκλωμα και το βάζει να παίξει μία "
        "παρτίδα απέναντι στις άλλες αρχιτεκτονικές, όλες παγωμένες. Έτσι βλέπεις αν αυτό που "
        "εξελίχθηκε στέκεται και έξω από τον κόσμο στον οποίο εξελίχθηκε, που είναι και το "
        "πραγματικό ερώτημα.",
        "At the end of evolution it takes the best circuit and has it play one match against "
        "the other architectures, all frozen. That shows whether what evolved also holds up "
        "outside the world it evolved in, which is the real question."),
}


#: Page-level explanations: the paragraphs that sit under a heading or beside a panel and say what
#: the reader is looking at. Same `(greek, english)` shape as `T` and `H`, kept here for the same
#: reason -- prose left in a template is prose in one language that nothing notices going stale.
N: dict[str, tuple[str, str]] = {
    # ---- match form ----------------------------------------------------------------------
    "note.congestion_optional": (
        "Όλα τα παρακάτω είναι προαιρετικά. Άσε ένα πεδίο κενό και κρατά την τιμή της έτοιμης "
        "ρύθμισης που διάλεξες, που για τις δημοσιευμένες είναι ακριβώς η ρύθμιση που έτρεξαν "
        "τα αντίστοιχα άρθρα.",
        "Everything below is optional. Leave a field blank and it keeps whatever the preset you "
        "chose uses, which for the published ones is exactly the configuration those papers "
        "ran."),
    "note.positions_explained": (
        "Με 2 θέσεις μία ταυτόχρονη κίνηση κρίνει τα πάντα, χωρίς να προλάβει κανείς να δει "
        "τους άλλους. Με 3 ή περισσότερες υπάρχει ενδιάμεσο βήμα, οπότε η κίνηση κάποιου "
        "γίνεται ορατή πριν δεσμευτεί, και μόνο τότε μπορεί να προκύψει σιωπηλή συνεννόηση.",
        "At 2 positions a single simultaneous move decides everything, with nobody able to see "
        "the others first. At 3 or more there is an intermediate step, so somebody's approach "
        "becomes visible before they commit, and only then can silent coordination emerge."),
    "note.zero_floor": (
        "Ο δημοσιευμένος κύριος κανόνας κρατά την καθολική σύγκρουση ακριβώς στο μηδέν, ώστε "
        "η πλήρης αποτυχία συνεννόησης να μην μπορεί να καταγραφεί ως μερική επιτυχία.",
        "The published main rule keeps universal collision at exactly zero, so that a total "
        "failure to coordinate cannot register as partial success."),
    "note.total_agents": ("Σύνολο παικτών:", "Total players:"),
    "note.squad_presets": (
        "Έτοιμες παρέες: αντικαθιστούν όλη τη σύνθεση με ένα κλικ, και μετά μπορείς να την "
        "πειράξεις όπως θες.",
        "Ready-made squads: one click replaces the whole roster, and you can edit it afterwards."),

    # ---- sweep ---------------------------------------------------------------------------
    "note.sweep_intro": (
        "Διάλεξε μια παράμετρο και ένα εύρος τιμών. Κάθε τιμή γίνεται ξεχωριστό πείραμα, όλα "
        "τρέχουν ταυτόχρονα στους πυρήνες αυτού του υπολογιστή, και κάθε μέτρηση σχεδιάζεται "
        "απέναντι στην παράμετρο μόλις τα αποτελέσματα φτάνουν.",
        "Pick a parameter and a range. Each value becomes its own experiment, they all run at "
        "the same time across this machine's cores, and every measure is drawn against the "
        "parameter as the results land."),
    "note.sweep_workers": (
        "Δεν προσφέρεται σκόπιμα ένας εργάτης ανά πυρήνα. Ο πυρήνας που μένει ελεύθερος είναι "
        "αυτός που κρατά τη σελίδα να απαντά όσο τρέχει η σάρωση.",
        "One worker per core is deliberately not offered. The spare core is what keeps this "
        "page answering while the sweep runs."),
    "note.sweep_starting": ("ξεκινά", "starting"),
    "note.sweep_chart": (
        "Κάθε κουκκίδα είναι ένα τρέξιμο. Η γραμμή ενώνει τον μέσο όρο σε κάθε τιμή. "
        "Σχεδιάζονται και τα δύο επειδή η γραμμή μόνη της κρύβει το πόσο διαφώνησαν μεταξύ "
        "τους τα τρεξίματα, που συχνά είναι πιο σημαντικό από τον ίδιο τον μέσο όρο.",
        "Each dot is one run. The line joins the mean at each value. Both are drawn because the "
        "line alone hides how much the runs disagreed with each other, which is often more "
        "important than the mean itself."),

    # ---- coverage ------------------------------------------------------------------------
    "note.coverage_board": (
        "Κάθε παιχνίδι και κάθε είδος πράκτορα για το οποίο υπάρχει έστω μία καταγραφή, και τα "
        "τετράγωνα που είναι ακόμη άδεια. Τίποτα εδώ δεν είναι γραμμένο στο χέρι: ο πίνακας "
        "διαβάζεται από το αρχείο πειραμάτων, οπότε δείχνει τι έχεις πράγματι τρέξει και όχι "
        "τι νομίζεις ότι έτρεξες.",
        "Every game and agent kind the ledger holds at least one record for, and the squares "
        "that are still empty. Nothing here is written by hand: the table is read from the "
        "ledger, so it shows what you have actually run rather than what you think you ran."),
    "note.coverage_unit": (
        "Ο αριθμός μετρά τρεξίματα, και η γραμμή από κάτω λέει σε πόσα διαφορετικά πειράματα "
        "αντιστοιχούν. Δέκα τρεξίματα ενός πειράματος είναι ένα σημείο επαναλαμβανόμενο δέκα "
        "φορές, όχι δέκα σημεία.",
        "The count is runs, and the line beneath says how many distinct experiments those runs "
        "are. Ten runs of one experiment is one point repeated ten times, not ten points."),
    "note.coverage_empty": (
        "Το αρχείο πειραμάτων είναι άδειο. Τρέξε μια παρτίδα και θα εμφανιστεί εδώ.",
        "The ledger is empty. Run a match and it will appear here."),
    "note.corr_intro": (
        "Κάθε ζεύγος από τα παρακάτω πεδία, πάνω σε όλα όσα έχουν καταγραφεί. Είναι εργαλείο "
        "για να αποφασίσεις πού αξίζει να κοιτάξεις μετά, όχι μέρος για να διαβάσεις ένα "
        "εύρημα: οι συσχετίσεις εδώ δεν προέκυψαν από σχεδιασμένο πείραμα.",
        "Every pair of the fields below, over everything recorded. It is a tool for deciding "
        "where to look next, not a place to read a finding: the correlations here did not come "
        "from a designed experiment."),
    "note.corr_legend": (
        "Το μπλε είναι θετική συσχέτιση, το κόκκινο αρνητική, και όσο πιο βαθύ το χρώμα τόσο "
        "ισχυρότερη. Ο χρυσός δακτύλιος σημαίνει ότι το ζεύγος επιβιώνει της διόρθωσης για το "
        "πλήθος των ελέγχων, δηλαδή δεν εξηγείται από το ότι κοιτάξαμε πολλά ζεύγη μαζί.",
        "Blue is a positive correlation, red negative, and the deeper the colour the stronger "
        "it is. A gold ring means the pair survives the correction for how many tests were run, "
        "that is, it is not explained by having looked at many pairs at once."),
    "note.corr_nothing": ("Δεν υπάρχει ακόμη τίποτα να σχεδιαστεί.", "Nothing to draw yet."),

    # ---- space map -----------------------------------------------------------------------
    "note.spacemap_intro": (
        "Ο κάνναβος των πειραμάτων που δήλωσες για το Public Goods Game, ελεγμένος απέναντι στο "
        "αρχείο: ποια κελιά έχουν ήδη τρέξει και ποια λείπουν ακόμη.",
        "The grid of experiments you declared for Public Goods, checked against the ledger: "
        "which cells have already run and which are still missing."),
    "note.spacemap_cap": (
        "τρέχουν το ένα μετά το άλλο· τα κελιά με DQN στους 3000 γύρους είναι τα πιο αργά",
        "they run one after another; DQN cells at 3000 rounds take the longest"),
    "note.spacemap_cells": (
        "Κάθε κελί αθροίζει πάνω στους γύρους και στους σπόρους του καννάβου: καλυμμένα "
        "τρεξίματα προς συνολικούς έγκυρους συνδυασμούς. Πράσινο σημαίνει πλήρως καλυμμένο, "
        "πορτοκαλί μερικώς.",
        "Each cell aggregates over the grid's rounds and seeds: covered runs against total "
        "valid combinations. Green means fully covered, amber partly."),

    # ---- analytics -----------------------------------------------------------------------
    "note.analytics_intro": (
        "Ψάξε σε κάθε τρέξιμο που έχει καταγραφεί ποτέ. Τα φίλτρα συνδυάζονται με ΚΑΙ, δηλαδή "
        "πρέπει να ισχύουν όλα μαζί. Ένα φίλτρο πάνω σε παράμετρο αρχιτεκτονικής ικανοποιείται "
        "αν το πληροί έστω ένας πράκτορας εκείνου του είδους μέσα στην παρτίδα.",
        "Search every run ever recorded. Filters combine with AND, so they must all hold at "
        "once. A filter on an architecture's parameter is satisfied if at least one agent of "
        "that kind in the match meets it."),
    "note.analytics_pearson": (
        "Συσχέτιση πάνω σε καταγεγραμμένα πειράματα που δεν σχεδιάστηκαν γι' αυτή την "
        "ερώτηση, οπότε διάβασέ την ως ένδειξη για το πού να κοιτάξεις και όχι ως αποτέλεσμα.",
        "A correlation over recorded experiments that were not designed for this question, so "
        "read it as a hint about where to look rather than as a result."),
    "note.analytics_nocorr": (
        "Δεν εμφανίζεται συσχέτιση: χρειάζονται τουλάχιστον 3 τρεξίματα που να ταιριάζουν, με "
        "διακύμανση και στους δύο άξονες.",
        "No correlation shown: it needs at least 3 matching runs, with variation on both axes."),
    "note.analytics_svg": (
        "πραγματικό διανυσματικό αρχείο και όχι στιγμιότυπο οθόνης, κατάλληλο να μπει "
        "απευθείας σε εικόνα άρθρου χωρίς να θολώσει.",
        "a real vector file rather than a screenshot, suitable to drop straight into a paper "
        "figure without going blurry."),
    "note.analytics_noaxes": (
        "Κανένα ζεύγος τρεξιμάτων δεν έχει και τους δύο άξονες που διάλεξες, οπότε δεν υπάρχει "
        "τίποτα να σχεδιαστεί.",
        "No two matching runs share both of the axes you chose, so there is nothing to plot."),
    "note.analytics_tick": (
        "Τσέκαρε δύο ή περισσότερα για σύγκριση, ή άνοιξε ένα για να δεις το τρέξιμο όπως "
        "αποθηκεύτηκε.",
        "Tick two or more to compare, or open one to see the run as it was stored."),
    "note.analytics_none": (
        "Κανένα καταγεγραμμένο τρέξιμο δεν ταιριάζει ακόμη με αυτά τα φίλτρα.",
        "No recorded run matches these filters yet."),

    # ---- meta-analysis -------------------------------------------------------------------
    "note.meta_onechain": (
        "Μία αλυσίδα ανά εγκατάσταση. Μια αλυσίδα που εισάγεις κρατιέται δίπλα στη δική σου και "
        "δεν ανακατεύεται ποτέ μέσα της, ώστε δύο εγκαταστάσεις να μη γράφουν ποτέ στο ίδιο "
        "πράγμα και να μη χρειάζεται κανένας μηχανισμός συμφωνίας.",
        "One chain per installation. A chain you import is kept beside your own and never "
        "merged into it, so two installations never write to the same thing and no agreement "
        "mechanism is needed."),
    "note.meta_keychannel": (
        "Το κλειδί πρέπει να φτάσει σε σένα από αλλού και όχι μέσα από το ίδιο το αρχείο. Ένα "
        "αρχείο που εγγυάται για τον εαυτό του αποδεικνύει μόνο ότι όποιος το έφτιαξε είχε το "
        "κλειδί του, που δεν σου λέει τίποτα για το ποιος είναι.",
        "The key has to reach you some other way than inside the file itself. A bundle that "
        "vouches for itself proves only that whoever built it had its key, which tells you "
        "nothing about who they are."),
    "note.meta_export": (
        "Περιέχει την αλυσίδα, τα αντικείμενα στα οποία δείχνουν οι εγγραφές της, και το "
        "δημόσιο κλειδί. Ποτέ το ιδιωτικό κλειδί.",
        "It carries the chain, the objects its records point at, and the public key. Never the "
        "private key."),
    "note.meta_rows": (
        "Κάθε γραμμή είναι ένα πείραμα: η κουκκίδα είναι ο μέσος όρος, η μπάρα το διάστημα "
        "εμπιστοσύνης 95%, και οι μικρές κουκκίδες τα επιμέρους τρεξίματα. Σχεδιάζονται και τα "
        "τρία γιατί ο μέσος όρος μόνος του δεν λέει πόσο συμφώνησαν τα τρεξίματα μεταξύ τους.",
        "Each row is one experiment: the dot is the mean, the bar is the 95% interval, and the "
        "small dots are the individual runs. All three are drawn because the mean alone does "
        "not say how much the runs agreed with each other."),
    "note.meta_single": (
        "Ένα πείραμα με ένα μόνο τρέξιμο αναφέρει μέσο όρο και κανένα διάστημα. Ένα μηδενικό "
        "διάστημα θα ήταν ισχυρισμός για τη διακύμανση βγαλμένος από δεδομένα που δεν μπορούν "
        "να τον στηρίξουν.",
        "An experiment with one run reports its mean and no interval. A zero interval would be "
        "a claim about variation drawn from data that cannot support it."),
    "note.meta_nometric": (
        "Κανένα καταγεγραμμένο τρέξιμο δεν αναφέρει ακόμη αυτό το μέτρο.",
        "No recorded run reports this measure yet."),
    "note.meta_question": (
        "Αυτή είναι η ερώτηση για την οποία υπάρχει το αρχείο πειραμάτων: δίνει το ίδιο "
        "δηλωμένο πείραμα την ίδια απάντηση και κάπου αλλού;",
        "This is the question the ledger exists for: does the same declared experiment give the "
        "same answer somewhere else?"),
    "note.meta_singlesource": (
        "Οι πηγές που στηρίζονται σε ένα μόνο τρέξιμο παραμένουν στον πίνακα αλλά εξαιρούνται "
        "από τους ελέγχους ετερογένειας, αφού μια εκτίμηση χωρίς διακύμανση δεν μπορεί να "
        "συνεισφέρει σε αυτούς.",
        "Sources whose mean rests on a single run stay in the table but are excluded from the "
        "heterogeneity tests, since a point estimate with no spread cannot contribute to them."),
    "note.download_both": ("Κατέβασε SVG · Κατέβασε CSV", "Download SVG · Download CSV"),
    "note.svg_link": ("Κατέβασε το γράφημα ως SVG", "Download this chart as an SVG"),
    "note.ollama_absent": (
        "Το Ollama δεν απαντά, οπότε δεν μπορώ να δω τι μοντέλα έχεις. Ξεκίνα το με "
        "ollama serve και ανανέωσε τη σελίδα για να εμφανιστούν.",
        "Ollama is not answering, so the models you have cannot be listed. Start it with "
        "ollama serve and reload the page to see them."),
    "note.ollama_empty": (
        "Το Ollama τρέχει αλλά δεν έχει κανένα μοντέλο. Κατέβασε ένα με "
        "ollama pull llama3.2:1b και ανανέωσε τη σελίδα.",
        "Ollama is running but has no models. Pull one with ollama pull llama3.2:1b and reload "
        "the page."),
    # ---- results page --------------------------------------------------------------------
    "note.genome_cid": (
        "Αντίγραψε αυτόν τον κωδικό στο πεδίο «Κωδικός εξελιγμένου γονιδιώματος» μιας σειράς "
        "markov_brain σε επόμενο τρέξιμο, για να ξαναπαίξει αυτό ακριβώς το κύκλωμα αντί για "
        "ένα καινούργιο τυχαίο.",
        "Copy this identifier into the Evolved genome CID field of a markov_brain row on a "
        "future run, so that exact circuit plays again instead of a fresh random one."),
    "note.alt_episodes_tail": (
        "Σε όλα τα μέτρα, όσο μεγαλύτερο τόσο καλύτερο.",
        "Higher is better throughout."),
    "note.alt_compare": (
        "Σύγκρινε τις στήλες δικαιοσύνης με τα CALT και AALT. Η δικαιοσύνη που κοιτά μόνο το "
        "τελικό αποτέλεσμα μπορεί να δείχνει σχεδόν 1, ενώ τα μέτρα εναλλαγής δείχνουν σχεδόν "
        "0. Αυτή ακριβώς είναι η περίπτωση για την οποία υπάρχουν αυτά τα μέτρα: όλοι πήραν "
        "περίπου τα ίδια, αλλά κανείς δεν πήρε ποτέ σειρά.",
        "Compare the fairness columns against CALT and AALT. Fairness that looks only at the "
        "final outcome can sit near 1 while the alternation measures sit near 0. That is "
        "exactly the case these measures exist to expose: everyone got a similar amount, but "
        "nobody ever took a turn."),
    "note.te_method": (
        "Κάθε ζεύγος παρακάτω ελέγχεται απέναντι σε τυχαιοποιημένες εκδοχές των ίδιων "
        "δεδομένων. Μόνο όσα ζεύγη περνούν τον έλεγχο μετρούν στο συγκεντρωτικό νούμερο "
        "παραπάνω. Μια μη μηδενική τιμή με μεγάλο p εδώ δεν είναι ένδειξη πραγματικής "
        "σύνδεσης: μπορεί να είναι θόρυβος από λίγα δεδομένα, ή να οφείλεται στο ότι και οι "
        "δύο παίκτες αντιδρούν στο ίδιο κοινό σήμα.",
        "Every pair below is tested against randomised versions of the same data. Only pairs "
        "that pass count toward the summary figure above. A non-zero value with a large p here "
        "is not evidence of a real link: it can be noise from too little data, or it can come "
        "from both players reacting to the same shared signal."),
    "note.graphs_explained": (
        "Δύο δίκτυα πάνω στην ίδια παρτίδα. Στο πρώτο, δύο παίκτες συνδέονται όσο πιο συχνά "
        "συνεργάστηκαν ταυτόχρονα, δηλαδή είναι κυριολεκτικά το ποιος-συνεργάζεται-με-ποιον. "
        "Στο δεύτερο, υπάρχει βέλος μόνο εκεί όπου η επιρροή παραπάνω πέρασε τον στατιστικό "
        "έλεγχο, οπότε είναι απλώς η εικόνα των σημαντικών συνδέσεων και όχι νέος υπολογισμός.",
        "Two networks over the same match. In the first, two players are linked the more often "
        "they cooperated at the same time, so it is literally the who-cooperates-with-whom "
        "network. In the second, an arrow exists only where the influence above passed its "
        "statistical test, so it is a picture of the significant links rather than a new "
        "computation."),
    "note.no_influence": (
        "Καμία σύνδεση επιρροής δεν πέρασε τον στατιστικό έλεγχο σε αυτή την παρτίδα, οπότε το "
        "δίκτυο επιρροής δεν έχει βέλη. Αυτό είναι εύρημα, όχι αποτυχία: σημαίνει ότι ο "
        "καθένας έπαιζε ουσιαστικά ανεξάρτητα από τους άλλους.",
        "No influence link passed the statistical test in this match, so the influence network "
        "has no arrows. That is a finding, not a failure: it means each player was essentially "
        "acting independently of the others."),
    "note.no_prior": (
        "Δεν υπάρχουν ακόμη προηγούμενα πειράματα στο αρχείο για να γίνει σύγκριση.",
        "There are no earlier experiments in the ledger to compare against yet."),

    # ---- a stored run --------------------------------------------------------------------
    "note.not_byte_identical": (
        "Αυτό το τρέξιμο δεν υπόσχεται ότι θα βγάλει ακριβώς τα ίδια αποτελέσματα αν "
        "επαναληφθεί, εξαιτίας του:",
        "This run does not promise the same results if it is repeated, because of:"),
    "note.lineage": ("Προέλευση:", "Lineage:"),
    "note.log_missing": (
        "Η εγγραφή υπάρχει αλλά το αναλυτικό της αρχείο όχι. Αυτό συμβαίνει όταν έχει "
        "εισαχθεί αρχείο από άλλη εγκατάσταση χωρίς τα δεδομένα στα οποία δείχνει: η αλυσίδα "
        "εξακολουθεί να επαληθεύεται, απλώς οι λεπτομέρειες ανά γύρο δεν υπάρχουν για να "
        "δειχτούν.",
        "The record is here but its detailed log is not. That happens when a bundle was "
        "imported without the data its records point at: the chain still verifies, the "
        "round-by-round detail simply is not there to show."),
    "note.read_back_tail": (
        "Τίποτα από αυτά δεν ξαναϋπολογίστηκε ούτε ξανατρέχτηκε: διαβάστηκαν όπως είχαν "
        "αποθηκευτεί.",
        "None of this was recomputed or rerun: it was read back exactly as it was stored."),
    "note.withheld_in_words": (
        "Ένα μέτρο που αρνήθηκε να αναφέρει τον εαυτό του εμφανίζεται εδώ με λέξεις αντί για "
        "αριθμό, ακριβώς όπως αποθηκεύτηκε.",
        "A measure that declined to report itself appears here in words rather than as a "
        "number, exactly as it was stored."),
    "note.no_snapshots": (
        "Δεν υπάρχουν στιγμιότυπα των «μυαλών» σε αυτό το αρχείο. Γράφονται μόνο όταν το ζητά "
        "το τρέξιμο, οπότε η απουσία τους είναι ιδιότητα εκείνου του τρεξίματος και όχι αυτής "
        "της σελίδας.",
        "There are no snapshots of the agents' minds in this log. They are written only when a "
        "run asks for them, so their absence is a property of that run rather than of this "
        "page."),
    "note.distance_units": (
        "Η απόσταση μετριέται σε τυποποιημένες μονάδες, ζυγισμένες ώστε ο σχεδιασμός του "
        "πειράματος να μετρά περισσότερο από το μήκος του. Οι αριθμοί συγκρίνονται μεταξύ τους "
        "μέσα σε αυτή τη λίστα, όχι με αριθμούς από άλλο αρχείο.",
        "Distance is measured in standardised units, weighted so the experiment's design counts "
        "for more than its length. The numbers compare against each other within this list, not "
        "against numbers from another ledger."),
    "note.nothing_comparable": (
        "Δεν υπάρχει ακόμη κάτι συγκρίσιμο. Τα τρεξίματα που καταγράφηκαν πριν προστεθεί ένα "
        "νέο είδος πράκτορα περιγράφονται με λιγότερα στοιχεία, οπότε παραλείπονται αντί να "
        "συγκριθούν πάνω σε διαφορετικές βάσεις.",
        "Nothing comparable yet. Runs recorded before a new agent kind was added are described "
        "with fewer fields, so they are skipped rather than compared on different bases."),

    # ---- comparing runs ------------------------------------------------------------------
    "note.compare_hashes": (
        "Τα τρεξίματα που μοιράζονται τον ίδιο κωδικό σχεδιασμού διαφέρουν μόνο σε μήκος ή σε "
        "τυχαιότητα. Αν οι κωδικοί διαφέρουν, πρόκειται για διαφορετικά πειράματα, και η "
        "απόσταση ανάμεσα στα νούμερά τους περιλαμβάνει και αυτή τη διαφορά.",
        "Runs sharing a design identifier differ only in length or randomness. If the "
        "identifiers differ they are different experiments, and the gap between their numbers "
        "includes that difference too."),
    "note.compare_none": (
        "Δεν έχει επιλεγεί τίποτα. Τσέκαρε δύο ή περισσότερα τρεξίματα στη σελίδα Αναλύσεις "
        "και πάτησε σύγκριση.",
        "Nothing selected. Tick two or more runs on the Analytics page and choose compare."),
    "note.compare_withheld": (
        "Ένα μέτρο που κάποια τρεξίματα αρνήθηκαν να αναφέρουν δείχνει τον λόγο αντί για "
        "αριθμό, και η διαφορά τα αγνοεί αντί να θεωρήσει την απουσία ως τιμή.",
        "A measure some runs declined to report shows its reason rather than a number, and the "
        "spread ignores those rather than treating an absence as a value."),

    # ---- evolution -----------------------------------------------------------------------
    "note.evolve_intro": (
        "Αυτό δεν τρέχει ανά γύρο μέσα σε μία παρτίδα. Εξελίσσει έναν πληθυσμό κυκλωμάτων "
        "μέσα από πολλές παρτίδες, γενιά τη γενιά, που είναι εντελώς άλλη χρονική κλίμακα από "
        "τη σελίδα της Παρτίδας. Όταν τελειώσει, μπορείς να ζητήσεις μια δίκαιη τελική "
        "αναμέτρηση: κάθε αρχιτεκτονική φτάνει στα όριά της χωριστά και με τον δικό της τρόπο, "
        "μετά παγώνουν όλες και παίζουν μία παρτίδα μαζί. Έτσι αποφεύγεται η σύγχυση του να "
        "μαθαίνουν όλοι ταυτόχρονα ο ένας απέναντι στον άλλον.",
        "This does not run per round inside a single match. It evolves a population of circuits "
        "across many matches, generation by generation, which is an entirely different "
        "timescale from the Match page. When it finishes you can ask for a fair final "
        "comparison: each architecture is brought to its limit separately and in its own "
        "natural way, then all of them are frozen and play one match together. That avoids the "
        "confusion of everyone learning against each other at the same time."),
    "note.mode2": (
        "Τρόπος 2, εξελιγμένο κύκλωμα σε κανονική παρτίδα: τρέξε αυτό, πάρε τον κωδικό του "
        "κυκλώματος από τα αποτελέσματα, και βάλ' τον στο πεδίο «Κωδικός εξελιγμένου "
        "γονιδιώματος» μιας σειράς markov_brain στη σελίδα της Παρτίδας. Θα παίξει δίπλα σε "
        "πράκτορες που μαθαίνουν κανονικά.",
        "Mode 2, an evolved circuit in an ordinary match: run this, take the circuit's "
        "identifier from the results, and put it into the Evolved genome CID field of a "
        "markov_brain row on the Match page. It will play alongside agents that learn "
        "normally."),
    "note.mode3": (
        "Τρόπος 3, ένα είδος μαθαίνει και τα υπόλοιπα μένουν σταθερά: γίνεται ήδη από τη "
        "σελίδα της Παρτίδας χωρίς τίποτα καινούργιο. Φτιάξε σύνθεση με μόνο το είδος που "
        "θέλεις να μαθαίνει, και βάλε κλασικές στρατηγικές ως σταθερό φόντο.",
        "Mode 3, one kind learns while the rest stay fixed: this is already possible from the "
        "Match page with nothing new. Build a roster with only the kind you want to learn, and "
        "add classic strategies as a fixed background."),
    "note.mode4": ("Ο τρόπος 4 έχει καταγραφεί, δεν υπάρχει εδώ", "Mode 4 is noted, not here"),
    "note.other_modes": ("Άλλοι τρόποι σύγκρισης", "Other comparison modes"),
    # short connective phrases that sit between template expressions
    "w.of":            ("από", "of"),
    "w.match_filters": ("καταγεγραμμένα τρεξίματα ταιριάζουν με τα τρέχοντα φίλτρα",
                        "recorded runs match the current filters"),
    "w.over_n":        ("πάνω σε", "over"),
    "w.matching_runs": ("τρεξίματα που ταιριάζουν", "matching runs"),
    "w.design":        ("πείραμα", "design"),
    "w.designs":       ("πειράματα", "designs"),
    "w.point":         ("σημείο", "point"),
    "w.points":        ("σημεία", "points"),
    "w.measures_rec":  ("μέτρα καταγεγραμμένα:", "measures recorded:"),
    "w.records_readable": ("εγγραφές αναγνώσιμες.", "records readable."),
    "w.grouped_by_design": (
        "Τα τρεξίματα ομαδοποιούνται ανά πείραμα: όσα ρωτούν την ίδια ερώτηση με διαφορετική "
        "τυχαιότητα μετρούν μαζί, ενώ όσα βαθμολογήθηκαν με διαφορετικές συμβάσεις μένουν "
        "χωριστά, γιατί τα νούμερά τους δεν είναι συγκρίσιμα.",
        "Runs are grouped by design: those asking the same question with different randomness "
        "are counted together, while those scored under different conventions are kept apart, "
        "because their numbers are not comparable."),
    "w.covered":       ("καλυμμένα", "covered"),
    "w.missing":       ("λείπουν", "missing"),
    "w.invalid":       ("άκυρα", "invalid"),
    "w.cells_total":   ("κελιά συνολικά", "cells total"),
    "w.just_ran":      ("μόλις έτρεξαν", "just ran"),
    "w.missing_cells": ("κελιά που έλειπαν", "missing cells"),
    "w.dl_svg":        ("Κατέβασε SVG", "Download SVG"),
    "w.dl_csv":        ("Κατέβασε CSV", "Download CSV"),
    "w.at_level":      ("σε επίπεδο", "at the"),
    "w.level_suffix":  ("", ""),
    "w.pair":          ("ζεύγος", "pair"),
    "w.pairs":         ("ζεύγη", "pairs"),
    "w.testable":      ("ελέγξιμα,", "testable,"),
    "w.surviving":     ("επιβιώνουν της διόρθωσης Benjamini-Hochberg σε",
                        "surviving Benjamini-Hochberg at"),
    "w.pair_needs":     ("Ένα ζεύγος χρειάζεται", "A pair needs"),
    "w.points_before":  ("σημεία πριν αναφερθεί καν συντελεστής, και η σημαντικότητα ελέγχεται "
                         "ανακατεύοντας τα δεδομένα",
                         "points before a coefficient is reported at all, and significance is "
                         "checked by shuffling the data"),
    "w.shuffles_so":    ("φορές, οπότε η μικρότερη τιμή p που μπορεί να προκύψει είναι",
                         "times, so the smallest p-value that can come out is"),
    "w.not_causation": (
        "Συσχέτιση πάνω σε καταγεγραμμένα πειράματα, όχι αιτιότητα: τα τρεξίματα διαφέρουν σε "
        "ό,τι δεν καθόρισαν τα φίλτρα.",
        "Correlation across recorded experiments, not causation: the runs differ in whatever "
        "the filters did not pin down."),
    "w.episodes_over": ("Σε", "Over"),
    "w.episodes_done": ("ολοκληρωμένα γυρίσματα:", "completed episodes:"),
    "w.won_alone":     ("κερδήθηκαν από έναν παίκτη μόνο του", "won by a single agent"),
    "w.collided":      ("κατέληξαν σε σύγκρουση", "ended in a collision"),
    "w.nobody_arrived": ("χωρίς να φτάσει κανείς", "with nobody arriving"),
    "w.agent":         ("πράκτορας:", "agent:"),
    "w.round":         ("γύρος", "round"),
    "w.needs_about":   ("χρειάζεται περίπου", "needs about"),
    "w.rounds_word":   ("γύρους", "rounds"),
    "w.reached":       ("το έφτασε", "reached"),
    "w.not_reached":   ("δεν το φτάνει με όσους γύρους έβαλες",
                        "not reached with the rounds you set"),
    "w.read_back_from": ("Διαβάστηκε από το αποθηκευμένο αρχείο,",
                         "Read back from the stored log,"),
    "w.across":        ("σε", "across"),
    "w.episodes_word": ("γυρίσματα", "episodes"),
    "w.extends":       ("επεκτείνει", "extends"),
    "w.replicates":    ("επαναλαμβάνει", "replicates"),
    "w.no_pair_survives": (
        "Κανένα ζεύγος δεν επιβιώνει της διόρθωσης. Με τόσα λίγα σημεία αυτό είναι το "
        "αναμενόμενο και όχι αρνητικό αποτέλεσμα: χρειάζονται περισσότερα τρεξίματα.",
        "No pair survives the correction. With this few points that is the expected outcome "
        "rather than a negative result: more runs are needed."),
    "note.mode4_tail": (
        ": εναλλασσόμενες φάσεις όπου κάποιοι μαθαίνουν και κάποιοι είναι παγωμένοι. "
        "Καταγράφηκε ως ερευνητικό ερώτημα για επόμενη εργασία, όχι ως λειτουργία εδώ.",
        ": alternating phases where some learn and some are frozen. It is recorded as a "
        "research question for later work, not as a feature here."),
    "field.compute_phi_evo": (
        "Υπολόγισε Φ και αιτιακή αυτονομία για το καλύτερο εξελιγμένο κύκλωμα (αργό)",
        "Compute Phi and causal autonomy for the best evolved circuit (slow)"),
    "note.fitness_explained": (
        "Η καταλληλότητα είναι ο μέσος όρος κέρδους από παρτίδες όπου τα κυκλώματα παίζουν "
        "μεταξύ τους. Στο Public Goods Game με πραγματικό δίλημμα, τόσο η θεωρία όσο και οι άλλες "
        "προσεγγίσεις προβλέπουν σύγκλιση κοντά στο 0, δηλαδή να μη δίνει κανείς τίποτα. "
        "Σύγκρινέ το με το πάνελ ισορροπίας Nash στη σελίδα της Παρτίδας.",
        "Fitness is the mean payoff from matches where the circuits play against each other. In "
        "Public Goods with a real dilemma, both theory and the other approaches predict "
        "convergence near 0, that is, nobody giving anything. Compare it against the Nash "
        "equilibrium panel on the Match page."),
    "note.best_genome_cid": (
        "Ο κωδικός του καλύτερου εξελιγμένου κυκλώματος. Χρησιμοποίησέ τον σε μια σειρά "
        "markov_brain στη σελίδα της Παρτίδας:",
        "The identifier of the best evolved circuit. Use it in a markov_brain row on the Match "
        "page:"),

    # ---- batch ---------------------------------------------------------------------------
    "note.batch_rows": (
        "Κάθε γραμμή είναι ξεχωριστό, πλήρως καταγεγραμμένο τρέξιμο: ίδια βασική ρύθμιση, ένα "
        "μόνο πεδίο αλλαγμένο. Άνοιξε οποιοδήποτε από τη σελίδα Αναλύσεις για να δεις τις "
        "λεπτομέρειές του.",
        "Each row is a separate, fully recorded run: the same base configuration with a single "
        "field changed. Open any of them from the Analytics page to see its detail."),
}


#: One entry per game the interface offers, keyed by the value its `<option>` carries. The game
#: selector shows the matching line the moment a game is picked, so the page explains whichever
#: game is chosen instead of carrying a list that goes stale.
#:
#: **A new game adds its line here in the same commit that adds the game.** `tests/test_i18n.py`
#: walks `_GAME_CHOICES` and fails on any value missing from this table, because the alternative
#: is a selector that quietly offers something nobody explained. The games' own names are proper
#: names and are never translated: the label on screen, the record in the ledger and the name in
#: the papers have to be the same string in every language.
GAMES: dict[str, tuple[str, str]] = {
    "public_goods": (
        "Κάθε γύρο ο καθένας διαλέγει αν θα συνεισφέρει μία μονάδα στο κοινό ταμείο. Το ταμείο "
        "πολλαπλασιάζεται και μοιράζεται ίσα σε όλους, ακόμη και σε όσους δεν έβαλαν τίποτα. "
        "Άρα το να κρατήσεις τη μονάδα σου συμφέρει πάντα ατομικά, ενώ αν την κρατήσουν όλοι "
        "βγαίνουν όλοι χαμένοι. Είναι η γενίκευση του διλήμματος του φυλακισμένου σε n παίκτες, "
        "και το κλασικό εύρημα είναι ότι η συνεργασία καταρρέει με τον χρόνο.",
        "Each round every player chooses whether to contribute one unit to a common pool. The "
        "pool is multiplied and shared equally among everyone, including those who contributed "
        "nothing. So keeping your unit always pays individually, while everyone keeping theirs "
        "leaves everyone worse off. It is the n-player generalisation of the prisoner's "
        "dilemma, and the classic finding is that cooperation collapses over time."),
    "congestion:hjg": (
        "Οι παίκτες κινούνται σε έναν διάδρομο προς ένα βραβείο που παίρνει μόνο όποιος φτάσει "
        "μόνος του. Αν φτάσουν πολλοί μαζί, μοιράζονται ελάχιστα ή τίποτα. Καμία σταθερή "
        "στρατηγική δεν είναι καλή εδώ: το συλλογικά σωστό είναι να παίρνουν σειρά, κάτι που "
        "απαιτεί μνήμη του ποιος κέρδισε την προηγούμενη φορά. Αυτή είναι η δημοσιευμένη κύρια "
        "ρύθμιση, με την ποινή να μοιράζεται σε όλο τον πληθυσμό.",
        "Players move along a corridor toward a prize that only someone arriving alone can "
        "take. If several arrive together they share almost nothing. No fixed strategy is good "
        "here: the collectively right answer is to take turns, which needs memory of who won "
        "last time. This is the published main configuration, with the penalty shared across "
        "the whole population."),
    "congestion:hjg_iqf": (
        "Το ίδιο με την κύρια ρύθμιση, αλλά η ποινή για ταυτόχρονη άφιξη μεγαλώνει πολύ πιο "
        "απότομα με το μέγεθος του πληθυσμού. Χρησιμεύει για να δεις αν ένα εύρημα κρατά όταν "
        "η σύγκρουση κοστίζει δυσανάλογα περισσότερο. Δημοσιευμένη ρύθμιση.",
        "The same as the main configuration, but the penalty for arriving together grows far "
        "more steeply with population size. It is there to test whether a finding holds when "
        "collision costs disproportionately more. A published configuration."),
    "congestion:hjg_k": (
        "Ίδιο παιχνίδι, αλλά η ποινή εξαρτάται από το πόσοι πράγματι έφτασαν μαζί μαζί σου και "
        "όχι από το μέγεθος όλου του πληθυσμού. Αυτή είναι η κυριολεκτική έννοια της "
        "congestion: όσο πιο γεμάτος ο δρόμος που διάλεξες, τόσο χειρότερα περνάς. "
        "Δημοσιευμένος έλεγχος ανθεκτικότητας.",
        "The same game, but the penalty depends on how many actually arrived alongside you "
        "rather than on the size of the whole population. This is congestion in the literal "
        "sense: the busier the route you chose, the worse you do. A published robustness "
        "check."),
    "congestion:hjg_memory": (
        "Ίδιο με την κύρια ρύθμιση, αλλά οι παίκτες βλέπουν και ποιος κέρδισε το προηγούμενο "
        "γύρισμα. Αυτή η μία επιπλέον πληροφορία είναι που κάνει το «παίρνουμε σειρά» κάτι που "
        "μπορεί να μαθευτεί: χωρίς αυτήν, κανείς δεν ξέρει σε ποιον έχει έρθει η σειρά.",
        "The same as the main configuration, but players also see who won the previous round. "
        "That one extra piece of information is what makes taking turns learnable: without it "
        "nobody knows whose turn it is."),
    "congestion:market_entry": (
        "Η εκδοχή μιας κίνησης: όλοι αποφασίζουν ταυτόχρονα αν θα μπουν, χωρίς να προλάβουν να "
        "δουν τι κάνουν οι άλλοι, και όποιος μπει σε γεμάτη αγορά χάνει. Επειδή δεν υπάρχει "
        "ενδιάμεσο βήμα, δεν υπάρχει και περιθώριο για σιωπηλή συνεννόηση, που το κάνει καλό "
        "σημείο σύγκρισης απέναντι στις δυναμικές εκδοχές.",
        "The one-move version: everyone decides simultaneously whether to enter, with no chance "
        "to see what the others are doing, and whoever enters a crowded market loses. Because "
        "there is no intermediate step there is no room for silent coordination either, which "
        "makes it a good comparison point against the dynamic versions."),
}


def pick(pair: Optional[tuple[str, str]], lang: str) -> Optional[str]:
    """The `lang` half of a `(greek, english)` pair, English for anything that is not Greek."""
    if pair is None:
        return None
    return pair[0] if lang == "el" else pair[1]


def t(key: str, lang: str = DEFAULT_LANG) -> str:
    """A visible string. Falls back to the key itself, which makes a gap obvious on the page."""
    return pick(T.get(key), lang) or key


def game(key: str, lang: str = DEFAULT_LANG) -> str:
    """What one game asks of its players, or an empty string for a game with no entry yet."""
    return pick(GAMES.get(key), lang) or ""


def n(key: str, lang: str = DEFAULT_LANG) -> str:
    """A page-level explanation. Falls back to the key, so a gap is visible rather than blank."""
    return pick(N.get(key), lang) or key


def h(key: str, lang: str = DEFAULT_LANG) -> Optional[str]:
    """The help line behind a ⓘ, or `None` where there is none, so the badge is simply not drawn."""
    return pick(H.get(key), lang)


def normalise(lang: Optional[str]) -> str:
    """Whatever arrived from a cookie or a query string, reduced to a language this module has."""
    return lang if lang in LANGS else DEFAULT_LANG
