BASIC_SMARTS = {
    "ketone": "[#6][CX3](=O)[#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "lactone": "[o][c](=O)",
    "carbamic_ester": "[NX3][CX3](=[OX1])[OX2H0]",
    "carbamate": "[NX3,NX4+][CX3](=[OX1])[OX1-]",
    "carboxylate": "C(=O)[O-]",
    "cyanamide": "[NX3][CX2]#[NX1]",
    "nitrile": "[CX2]#[NX1]",
    "ether": "[O;X2;H0;+0]([#6])[#6]",
    # Recursive queries keep the ionizing oxygen as the site atom while still
    # requiring its aromatic environment. This prevents aromatic-ring and
    # generic oxygen patterns from stealing the protonation center.
    "phenolate": "[O;X1;-1]-[c]",
    "alkoxide": "[O;X1;-1]-[C;X4]",
    # Each neutral N in an N-N single bond is a distinct candidate centre.
    # H0/H1 variants matter for substituted hydrazines; keeping the match to
    # one atom avoids one N suppressing the other during overlap resolution.
    "hydrazine": "[N;X3;H0,H1,H2;+0;$([N]-[N;X3;+0])]",
    "hydroxylamine": "[N;X3;H0,H1,H2;+0;$([N]-[O;X2;+0])]",
    "enamine": "[N;X3;+0][CX3]=[CX3]",
    "primary_amine": "[NX3;H2;!$([N][C](=O));!$([N+])][CX4]",
    "secondary_amine": "[NX3;H1;!$([N][C](=O));!$([N+])]([CX4])[CX4]",
    # Three-membered aziridine nitrogens are materially less basic than
    # acyclic amines and therefore require their own transition family.
    "aziridine": "[N;X3;+0;r3:1]1[#6][#6]1",
    # Formal-charge and oxide exclusions are essential: without them the N+
    # of a nitro group is falsely recognized as a tertiary amine.
    "tertiary_amine": "[N;X3;H0;+0;!$(N-C=O);!$([N](=O)=O);!$([N]-[S,P](=O))]",
    # Permanent charge: this is a descriptor, not an ammonium acid form.
    "quaternary_ammonium": "[N+;X4;H0:1]([#6])([#6])([#6])[#6]",
    "amide_primary": "[NX3;H2;!$([N+])][#6](=[OX1])[!#1]",
    "amide_secondary": "[NX3;H1;!$([N+])][#6](=[OX1])[!#1]",
    "amide_tertiary": "[NX3;H0;!$([N+])][#6](=[OX1])[!#1]",
    "2amide_anhydride": "NC(=O)NC=O",
    "amide_anhydride": "C(=O)N(C=O)",
    "conjugated_hydrazine": "[N;X3;H2;+0;$([N]-[N;X3;+0]-[#6;X3])]",
    "carbamide": "[#7X3][#6X3]([#7X3])=[OX1]",
    "hydroxyamide": "[CX3]=[NX3H]O",
    "hydroxyguanmidine": "[NX3][CX3](O)=[NX3H]",
    "n-hydroxyguanidine": "[NX3][CX3]=[NX3H]O",
    "guanidine": "[#7X3][#6X3]([#7X3])=[#7X2]",
    "amidine": "[#7X3][#6X3]=[#7X2]",
    "hydroxyamidine": "[O][#6X3]=[#7X2]",
    "aminoamidine": "[#6X3]([#7X3])=[#7X2][#7X3]",
    "n-hydroxyamidine": "[#7X3][#6X3]=[#7X2][O]",
    "isoamide": "[OX1][#6X3]=[#7X3]",
    "lactam": "[n][c](=O)",
    # Exclude both anilides and aryl sulfonamides.  Their nitrogens are
    # resonance-deactivated and must not be treated as aniline-like bases.
    "aniline": "[N;X3;+0;!$([N][C](=O));!$([N][S](=O)(=O));!$([N]-[O,N,S])][c]",
    "pyrazole": "c1cc[nH]n1",
    "pyrazolate": "[n-:1]1nccc1",
    "pyrazolate_alt": "n1[n-:1]ccc1",
    "pyridine": "n1ccccc1",
    "pyrimidine": "n1cnccc1",
    "pyrazine": "n1ccncc1",
    "pyrrole": "n1cccc1",
    "pyrazolidine": "n1nccc1",
    "imidazole": "n1ccnc1",
    "imidazolate": "[n-:1]1ccnc1",
    "imidazolate_alt": "n1cc[n-:1]c1",
    "123triazole": "c1c[nH+0,nH0+0][nH+0,nH0+0][nH+0,nH0+0]1",
    "124triazole": "c1[nH+0,nH0+0][nH+0,nH0+0]c[nH+0,nH0+0]1",
    "123triazolate": "c1c[n-][nH0+0][nH0+0]1",
    "123triazolate_alt2": "c1c[nH0+0][n-][nH0+0]1",
    "123triazolate_alt3": "c1c[nH0+0][nH0+0][n-]1",
    "124triazolate": "c1[n-][nH0+0]c[nH0+0]1",
    "124triazolate_alt2": "c1[nH0+0][n-]c[nH0+0]1",
    "124triazolate_alt3": "c1[nH0+0][nH0+0]c[n-]1",
    # Neutral tetrazole is the acid form of the aqueous
    # tetrazole/tetrazolate transition. Atom-map 1 identifies the N-H centre;
    # the enumerator subsequently retains both 1H/2H neutral tautomers.
    "tetrazole": "[nH;r5;$([nH]1~[#6]~[#7]~[#7]~[#7]1),$([nH]1~[#7]~[#6]~[#7]~[#7]1),$([nH]1~[#7]~[#7]~[#6]~[#7]1),$([nH]1~[#7]~[#7]~[#7]~[#6]1):1]",
    "tetrazolate": "[n-;r5;$([n-]1~[#6]~[#7]~[#7]~[#7]1),$([n-]1~[#7]~[#6]~[#7]~[#7]1),$([n-]1~[#7]~[#7]~[#6]~[#7]1),$([n-]1~[#7]~[#7]~[#7]~[#6]1):1]",
    "n_substituted_tetrazole": "[#6;r5]1~[#7;r5]~[#7;r5]~[#7;r5]~[#7;r5]1",
    "triazine": "c1ncncn1",
    "124triazine": "c1nncnc1",
    "pyridazine": "[n;+0]1[n;+0]cccc1",
    "benzimidazole": "c21ncnc1cccc2",
    "7-azaindole": "c21nccc1cccn2",
    "n-indole": "c1c2n(ccc1)ccn2",
    "indazole": "c1ccc2[nH+0,nH0+0][nH+0,nH0+0]cc2c1",
    "indazolate": "c1ccc2[n-][n;+0]cc2c1",
    "indazolate_alt": "c1ccc2[n;+0][n-]cc2c1",
    "pyridazinone": "c1ccc(nn1)=O",
    "n-indazole": "c1c2c(cnc1)ncn2",
    "nn-indazole": "n1c2c(cnc1)ncn2",
    "furan": "[o]1cccc1",
    "oxazole": "c1ncco1",
    "isoxazole": "c1nocc1",
    # Common mixed N/O and N/S five-membered rings. These are structural
    # descriptors until a particular Brønsted transition is enabled.
    "123_oxadiazole": "o1nncc1",
    "124_oxadiazole": "o1ncnc1",
    "125_oxadiazole": "o1nccn1",
    "134_oxadiazole": "o1cnnc1",
    "123_thiadiazole": "s1nncc1",
    "124_thiadiazole": "s1ncnc1",
    "125_thiadiazole": "s1nccn1",
    "134_thiadiazole": "s1cnnc1",
    "124-s2-triazole": "n1cnsc1",
    "124-s4-triazole": "s1cnnc1",
    "indole": "c12c(ccn2)cccc1",
    "Nindole": "c12c(ncn2)cccc1",
    "isoNindole": "c12c(cnn2)cccc1",
    "Sindole": "c12c(scn2)cccc1",
    "isoquinoline": "c1ccc2cnccc2c1",
    "quinoline": "c1ccc2cccnc2c1",
    # Match all three N atoms, rather than the attachment atom of a recursive
    # query. This is a resonance descriptor only in the aqueous 0-14 model.
    "azide": "[N;X1,X2;-1,+0]~[N+;X2]~[N;X1,X2;-1,+0]",
    "azo": "[NX2]=[NX2]",
    "n-hydroxy": "[NX3][OX2H]",
    "hydrazone": "[NX3][NX2]=[*]",
    "imine": "[CX3]=[NX2]",
    "imide": "[CX3](=[OX1])[NX3H][CX3](=[OX1])",
    "nitrite": "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",
    # Match the nitro nitrogen itself. The former pattern included the
    # attachment atom, causing overlap resolution to discard nitro beside an
    # aromatic heterocycle; it also allowed false amine/aniline assignments.
    "nitro": "[N;$([N+;X3](=[O;X1])[O-;X1]),$([N;X3](=[O;X1])=[O;X1])]",
    "nitroso": "[NX2]=[OX1]",
    # N+-O- is a base form: protonation occurs on the mapped oxide oxygen.
    # Aromatic and tetra-coordinate aliphatic N-oxides are kept separate so
    # their very different environments remain available to Stage 1.
    "n-oxide": "[O-:1]-[n+;r]",
    "amine_n_oxide": "[O-:1]-[N+;X4]",
    "123triazine-n-oxide": "c1cnnn1O",
    "hydroxy-1234riazine": "n1cnoc1",
    "thioide": "[SX1-]",
    "thiophenol": "[S;X2;H1]-[c]",
    "thioamide": "[NX3][#6]=[SX1]",
    "thioamideamide_anhydride": "[#6](=[O])[#7][#6](=O)[#16]",
    "thioamideamide": "[NX3][CX3]([NX3])=[SX1]",
    "thioamidethio": "[NX3][CX3]([SX2])=[SX1]",
    "thiophene": "c1ccsc1",
    "thiazole": "c1cnsc1",
    "isothiazole": "c1cncs1",
    "isoxazolidine": "o1nccc1",
    # Historical label retained for compatibility; this is specifically a
    # thiolate anion, not a neutral sulfide/thione sulfur.
    "sulfide": "[#16X1-]",
    # Match only sulfur and allow C/N substituents. A one-atom recursive
    # query prevents an adjacent aromatic rule from swallowing charge-
    # separated S+-O- representations during overlap resolution.
    "sulfoxide": "[S;$([S;X3;+1]([O-;X1])([!O])[!O]),$([S;X3;+0](=[O;X1])([!O])[!O])]",
    # N-centred structural/ionization candidate. The previous S-centred
    # recursive query found the group but left its nitrogen undescribed.
    "sulfonamide": "[N;X3;$([N]-[S;X4](=[O;X1])(=[O;X1])[!O]),$([N]-[S;X4;+2]([O-;X1])([O-;X1])[!O])]",
    "sulfonate": "[SX4](=[OX1])(=[OX1])[O-]",
    "sulfone": "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",
    "sulfamide": "O=S(=O)(N)C",
    "sulfamate": "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2][#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2][#6])]",
    "sulfamate_anion": "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX1-]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX1-])]",
    # Non-ionizing atom-coverage descriptors for heteroatoms attached to the
    # acid centre. Their own N-H transitions need separate evidence.
    "sulfamoyl_nitrogen": "[N;X3;$([N]-[S;X4](=[O;X1])(=[O;X1])[O;X1,X2]),$([N]-[S;X4;+2]([O-;X1])([O-;X1])[O;X1,X2])]",
    "phosphoramidate_nitrogen": "[N;X3;$([N]-[P;X4](=[O;X1])([O,N,#6])([O,N,#6]))]",
    # Non-ionizing context descriptor for a C-O-P ester linkage. Acidic and
    # anionic P-O sites are defined separately below as one-atom sites.
    "phosphate_ester": "[O;X2;H0;$([O](-[#6])-[P;X4](=[O;X1])([O,N,#6])([O,N,#6]))]",
    "phosphate_anion": "[O;X1;-1;$([O-]-[P;X4](=[O;X1])([O])([O]))]",
    "phosphonate_anion": "[O;X1;-1;$([O-]-[P;X4](=[O;X1])([O])([#6]))]",
    "phosphinate_anion": "[O;X1;-1;$([O-]-[P;X4](=[O;X1])([#6])([#6]))]",
    "phosphoramidate_anion": "[O;X1;-1;$([O-]-[P;X4](=[O;X1])([O,N,#6])([N]))]",
    "thioimidate": "[SX2][CX3]=[NX1]",
    "thioester": "[SX2][CX3]=[OX1]",
    "hydroxamic_acid": "[OX2H1][NX3][C]=O",
    "thiolocarbonate": "[NX3][CX3]([SX2])=[OX1]",
    "aromatic_thiosulfamide": "c1ccccc1S(=O)(=O)N",
}

ACIDIC_SMARTS = {
    "boric_acid": "B(O)(O)",
    # Classify by the H count on the saturated carbon, rather than demanding
    # carbon-only substituents. This covers methanol and heteroatom-substituted
    # alcohols/hemiaminals while retaining primary/secondary/tertiary labels.
    "primary_alcohol": "[O;X2;H1]-[C;X4;H2,H3]",
    "secondary_alcohol": "[O;X2;H1]-[C;X4;H1]",
    "tertiary_alcohol": "[O;X2;H1]-[C;X4;H0]",
    "protonated_alcohol": "[O;X3;H2;+1;$([O]-[#6])]",
    "protonated_thiol": "[S;X3;H2;+1;$([S]-[#6])]",
    "hydrazinium": "[N;X4;+1;H1,H2,H3;$([N]-[N;X3;+0])]",
    "hydroxylammonium": "[N;X4;+1;H1,H2,H3;$([N]-[O;X2;+0])]",
    "phenol": "[O;X2;H1]-[c]",
    "peroxide": "[OX2,OX1-][OX2,OX1-]",
    "carboxylic_acid": "C(=O)[OX2H1]",
    "percaboxylic_acid": "C(=O)[O][OH]",
    "sulfonic_acid": "S(=O)(=O)[OX2H1]",
    "carbamic_acid": "[NX3,NX4+][CX3](=[OX1])[OX2H]",
    # Each proton-bearing P-OH is an independent site. Recursive environment
    # constraints distinguish phosphate, phosphonate, phosphinate and
    # phosphoramidate chemistry without adding the whole P group to atom_set.
    "phosphoric_acid": "[O;X2;H1;$([O]-[P;X4](=[O;X1])([O])([O]))]",
    "phosphonic_acid": "[O;X2;H1;$([O]-[P;X4](=[O;X1])([O])([#6]))]",
    "phosphinic_acid": "[O;X2;H1;$([O]-[P;X4](=[O;X1])([#6])([#6]))]",
    "phosphoramidic_acid": "[O;X2;H1;$([O]-[P;X4](=[O;X1])([O,N,#6])([N]))]",
    "thiol": "[SX2H]",
    "sulfinic_acid": "[$([#16X3](=[OX1])[OX2H,OX1H0-]),$([#16X3+]([OX1-])[OX2H,OX1H0-])]",
    "sulfamic_acid": "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2H]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2H])]",
    "sulfenic_acid": "[#16X2][OX2H,OX1H0-]",
    "guanidinium": "[NX3][CX3]([NX3])=[NX3+]",
    "hydroxyguanidinium": "[NX3][CX3](O)=[NX3+]",
    "amidinium": "[#7X3][#6X3]=[#7X3+]",
    "amidenium": "[#6X3](=[OX1])[#7X3+]",
    "other_aniline": "[NX3H]-[c]",
    "other_anilinium": "[NX3H+]-[c]",
    "iminium": "[CX3]=[NX3+]",
    # Thermodynamic conjugate acid of pyrrole.  Protonation occurs at an
    # alpha carbon (C2), not by adding a second proton to the pyrrolic N.
    # The positive charge is written on N as one valid non-aromatic
    # resonance form of the C2-protonated pyrrolium ion.
    "pyrrolium": "[N+;R]1=[C;R][C;R]=[C;R][CH2;R]1",
    # Kept distinct because N-protonated pyrrole is a different microstate
    # and does not inherit the textbook C2-protonation pKaH.
    "pyrrolium_n": "[N+;R;H1,H2]1[C;R]=[C;R][C;R]=[C;R]1",
    "pyridinium": "[nH+]1ccccc1",
    "isoquinolinium": "c1ccc2c[nH+]ccc2c1",
    "quinolinium": "c1ccc2ccc[nH+]c2c1",
    "pyrimidinium": "[$([nH+]1cnccc1),$(n1c[nH+]ccc1)]",
    "pyrazinium": "[$([nH+]1ccncc1),$(n1cc[nH+]cc1)]",
    "pyridazinium": "[nH+:1]1[n;+0]cccc1",
    "benzimidazolium": "[nH+:1]1c2ccccc2[nH]c1",
    "oxazolium": "c1[nH+:1]cco1",
    "isoxazolium": "c1[nH+:1]occ1",
    "thiazolium": "c1c[nH+:1]sc1",
    "isothiazolium": "c1c[nH+:1]cs1",
    "124triazolium": "c1[nH+][nH+0,nH0+0]c[nH+0,nH0+0]1",
    "124triazolium_alt2": "c1[nH+0,nH0+0][nH+]c[nH+0,nH0+0]1",
    "124triazolium_alt3": "c1[nH+0,nH0+0][nH+0,nH0+0]c[nH+]1",
    "indazolium": "c1ccc2[nH+][n,nH]cc2c1",
    "indazolium_alt": "c1ccc2[n,nH][nH+]cc2c1",
    "pyrazolium": "c1c[nH+][nH+0]c1",
    "pyrazolium_alt": "c1c[nH+0][nH+]c1",
    "triazinium": "c1ncncn[nH+]1",
    "tetrazolium": "c1nn[nH+]n1",
    "imidazolium": "n1cc[nH+]c1",
    "aryl_ammonium": "[N;X4;+1;H3]-[c]",
    "primary_ammonium": "[N;X4;+1;H3]-[#6]",
    "secondary_ammonium": "[N;X4;+1;H2]([#6])[#6]",
    "aziridinium": "[N;X4;+1;r3:1]1[#6][#6]1",
    "tertiary_ammonium": "[N;X4;+1;H1]([#6])([#6])[#6]",
    "protonated_n_oxide": "[O;X2;H1;+0:1]-[n+;r]",
    "protonated_amine_n_oxide": "[O;X2;H1;+0:1]-[N+;X4]",
}

# COMPILED LITERATURE pKa VALUES
# ACIDIC_PKA: pKa for loss of a proton from the acid form.
# For true acids (carboxylic_acid, phenol, …) this is the standard pKa.
# For conjugate acids of bases (ammonium, pyridinium, …) this is pKaH —
# the deprotonation equilibrium of the protonated base, which is the
# value typically reported in SDF files under the "pKa" label.
ACIDIC_PKA = { # pKa of the acid form (= pKaH for conjugate acids of bases)
    "primary_alcohol": 16,
    "secondary_alcohol": 16,
    "tertiary_alcohol": 16,
    "phenol": 10,
    "peroxide": 12,
    "carboxylic_acid": 4,
    "phosphoric_acid": 2,
    "thiol": 10,
    "sulfinic_acid": 2,
    "sulfamic_acid": 0,
    "sulfenic_acid": 0,
    "sulfonic_acid": -1,
    "amide_primary": 15,
    "amidinium": 12,        # N-H deprotonation, NOT C-H acidity
    "iminium": 7,            # C=NH₂⁺ → C=NH + H⁺
    "imidazolium": 7,
    "pyridinium": 5,
    "quinolinium": 4.9,
    "isoquinolinium": 5.4,
    "pyrimidinium": 1,       # weak base, pKaH ≈ 1.3
    "pyrazinium": 1,         # weak base, pKaH ≈ 0.6
    "pyrrolium": -3.8,        # C2-protonated thermodynamic conjugate acid
    "pyrazolium": 3,          # pKaH ≈ 2.5
    "triazinium": 2,          # pKaH ≈ 2
    "primary_ammonium": 10,
    "secondary_ammonium": 11,
    "aziridinium": 8.0,
    "tertiary_ammonium": 11,
    "aryl_ammonium": 5,
}

# https://organicchemistrydata.org/hansreich/resources/pka/#gsc.tab=0
# NOTE: These are C−H / N−H acidity values for the *neutral* base forms
# (pKa for losing a proton from the neutral species), NOT the pKaH values
# for the conjugate acids.  For the biologically relevant protonation
# equilibrium (e.g. R-NH₃⁺ ⇌ R-NH₂ + H⁺), see the corresponding
# conjugate-acid entries in ACIDIC_PKA (e.g. primary_ammonium → 9).
BASIC_PKA = { # C-H / N-H acidity of the neutral base form (NOT pKaH)
    "ketone": 20,
    "aldehyde": 17,
    "amide_secondary": 17,
    "ester": 25,
    "lactone": 25,
    "ether": 30,
    "primary_amine": 33,
    "secondary_amine": 34,
    "aziridine": 34,
    "tertiary_amine": 36,
    "amide_primary": 17,
    "lactam": 17,
    "aniline": 30,
    "imidazole": 14,
    "pyrazole": 14,
    "pyridine": 30,
    "quinoline": 30,
    "isoquinoline": 30,
    "pyrimidine": 30,
    "pyrazine": 30,
    "pyrrole": 24,
    "triazine": 20,
    "hydrazone": 30,
    "imine": 30,
    "imide": 30,
    "indole": 21,
    "thioamide": 30,
    "sulfone": 30,
    "sulfonamide": 30,
    "sulfamate": 30,
    "nitrite": 30,
    "n-oxide": 30,
    "sulfide": 30,
    "sulfoxide": 30,
    "sulfonate": 30,
    # this one makes me think because I need to recognise you but you only have 1 deprotonation
    "carboxylate": 30,
    "carbamate": 30,
    "carbamic_ester": 30,
    "cyanamide": 30,
    "enamine": 30,
    "thiol": 30,
    # if you include the neighboring CH to lose H, pKa 10
    "azide": 30,
    "nitro": 30,
    "azo": 30,
    "nitroso": 30,
}
