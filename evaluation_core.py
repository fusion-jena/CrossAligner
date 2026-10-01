from __future__ import annotations

from pathlib import Path
import hashlib, json, math, re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rdflib import Graph, RDF, RDFS, OWL, URIRef, Namespace

PAIR_ORDER = ["mesh_ons", "occo_mesh", "occo_ons"]
PAIR_LABELS = {"mesh_ons":"MeSH–ONS", "occo_mesh":"OccO–MeSH", "occo_ons":"OccO–ONS"}
PAIR_PREFIX = {"mesh_ons":"MO", "occo_mesh":"OM", "occo_ons":"OO"}
PAIR_ONTOLOGIES = {"mesh_ons":("mesh","ons"), "occo_mesh":("occo","mesh"), "occo_ons":("occo","ons")}
MODEL_ORDER = ["Llama", "Qwen"]
EXAMPLE_ORDER = ["without", "with"]
DIRECTION_ORDER = ["1-2", "2-1"]
STAGE2_MODEL = "Llama-3.3-70B-Instruct"
COLORS = {
    "semantic":"#4C78A8", "structural":"#59A14F", "repair":"#F28E2B",
    "robustness":"#B279A2", "stage1":"#E15759", "llama":"#4C78A8",
    "qwen":"#F28E2B", "neutral":"#6B7280",
}

SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
OBOINOWL = Namespace("http://www.geneontology.org/formats/oboInOwl#")
IAO_DEF = URIRef("http://purl.obolibrary.org/obo/IAO_0000115")


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_tables(table_dir: Path | str, tables: dict[str, pd.DataFrame]) -> None:
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(table_dir / f"{name}.csv", index=False)


def norm_label(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s


def pair_key(source, target) -> str:
    return f"{str(source)}||{str(target)}"


def _load_ontologies(root: Path):
    #files = {
     #   "mesh": root / "data/ontologies/mesh_food_subset_bot_minimal_minimal.owl",
      #  "occo": root / "data/ontologies/occo_food_service_bot_minimal_minimal.owl",
       # "ons": root / "data/ontologies/ons_bot_minimal_minimal.owl",
    #}

    files = {
        "mesh": root / "assets/food-onto/mesh.owl",
        "occo": root / "assets/food-onto/occo.owl",
        "ons": root / "assets/food-onto/ons.owl",
    }

    labels = {"mesh":"MeSH", "occo":"OccO", "ons":"ONS"}
    rows=[]; lookup={}; by_label={}; evidence=[]
    for ont, path in files.items():
        g=Graph(); g.parse(path)
        classes=sorted({c for c in g.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef)}, key=str)
        local=[]
        for c in classes:
            lbls=list(g.objects(c, RDFS.label)) + list(g.objects(c, SKOS.prefLabel))
            lbl=str(lbls[0]) if lbls else ""
            defs=list(g.objects(c, IAO_DEF)) + list(g.objects(c, SKOS.definition))
            syns=list(g.objects(c, OBOINOWL.hasExactSynonym)) + list(g.objects(c, SKOS.altLabel))
            parents=list(g.objects(c, RDFS.subClassOf))
            children=list(g.subjects(RDFS.subClassOf, c))
            equiv=list(g.objects(c, OWL.equivalentClass))
            rec={"ontology":ont,"ontology_label":labels[ont],"iri":str(c),"label":lbl,
                 "label_norm":norm_label(lbl),"has_label":bool(lbls),"has_definition":bool(defs),
                 "has_synonym":bool(syns),"has_parent":bool(parents),"has_child":bool(children),
                 "has_axiom":bool(parents or equiv)}
            rows.append(rec); local.append(rec); lookup[str(c)]=rec
            if lbl:
                by_label[(ont,norm_label(lbl))]=rec
        norms=[r['label_norm'] for r in local if r['label_norm']]
        dup=set(x for x,n in Counter(norms).items() if n>1)
        n=len(local)
        evidence.append({
            "ontology":ont,"ontology_label":labels[ont],"classes":n,
            "label_coverage":sum(r['has_label'] for r in local)/n,
            "definition_coverage":sum(r['has_definition'] for r in local)/n,
            "synonym_coverage":sum(r['has_synonym'] for r in local)/n,
            "parent_coverage":sum(r['has_parent'] for r in local)/n,
            "child_coverage":sum(r['has_child'] for r in local)/n,
            "axiom_coverage":sum(r['has_axiom'] for r in local)/n,
            "duplicate_label_ambiguity":sum(r['label_norm'] in dup for r in local)/n,
        })
    return pd.DataFrame(rows), pd.DataFrame(evidence), lookup, by_label


GROUPED_TEXT_OVERRIDES = {('mesh_ons', 3, 'right_group'): 'diet by nutritonal composition;\n'
                                 'DASH diet;\n'
                                 'gluten free diet:\n'
                                 'ketogenic diet;\n'
                                 'low-carbohydrate, high-protien, high fat diet rural diet',
 ('mesh_ons', 3, 'property_labels_raw'): 'checked_for,\nhas,\ncharacterized_by,\nDefines,\ndefines_nutritional_composition_of',
 ('mesh_ons', 3, 'reasons'): '1. Looked for them before preparing a diet\n'
                             '2. they both describe the quality of a food\n'
                             '3. diet needs information about nutritions',
 ('mesh_ons', 3, 'evidence'): 'MeSH Nutritive Value concerns the nutrient contribution of food to diet. ONS diet by nutritional composition is explicitly '
                              'defined by nutritional presence or absence.\n'
                              'A diet is a collection of food types/items, while nutritive value indicates the nutrient contribution of food to a diet.',
 ('mesh_ons', 3, 'evidence_fields'): 'MeSH definition; hierarchy;\nONS labels; definitions; hierarchy.',
 ('mesh_ons', 4, 'reasons'): '1. The food material has a specific nutritive value.\n2. food material has nutritive value',
 ('mesh_ons', 4, 'evidence_fields'): 'MeSH definition; hierarchy;\nONS labels; definitions; hierarchy.',
 ('mesh_ons', 5, 'right_group'): 'Food product\nAnimal food product,\nanimal seafood product,\nfish food product,\nfungus food product,\nplant food product,',
 ('mesh_ons', 5, 'property_labels_raw'): 'has,\nDerives,\nDefine,\ngiven_to,\nassigned_to',
 ('mesh_ons', 5, 'reasons'): '1. assigned to the food products\n2. food product has nutritive value',
 ('mesh_ons', 5, 'evidence_fields'): 'MeSH definition; hierarchy;\nONS labels; definitions; hierarchy.',
 ('mesh_ons', 6, 'right_group'): 'Food product\nAnimal food product,\nanimal seafood product,\nfish food product,\nfungus food product,\nplant food product,',
 ('mesh_ons', 6, 'property_labels_raw'): 'affects,\nrisk,\nHas,\noccurs_in (RO)\nmonitor_for',
 ('mesh_ons', 6, 'reasons'): '1. Food contamination affects food products\n2. The food product has a certain contamination.\n3. all food can be contaminated',
 ('mesh_ons', 6, 'evidence_fields'): 'MeSH definition; hierarchy;\nONS labels; definitions; hierarchy.',
 ('mesh_ons', 7, 'property_labels_raw'): 'Has,\nhas_quality, (RO)\ncheck_by,\nassessed_by,\nevaluates,\nuses_food_label_information',
 ('mesh_ons', 7, 'reasons'): '1. Food material /food products check food quality\n'
                             '2. Food material /food products has food quality\n'
                             '3. Food products (including plant, animal, fish, and fungal products) are systematically evaluated and assessed for freshness, '
                             'safety, and quality attributes.',
 ('mesh_ons', 7, 'evidence'): 'Food quality definition: Ratings of the characteristics of food including flavor, appearance, nutritional content, and the '
                              'amount of microbial and chemical contamination.\n'
                              'Food material for humans and animals which is processed with the intention that it be consumable as a whole or added to other '
                              'food products.',
 ('mesh_ons', 7, 'evidence_fields'): 'MeSH definition; hierarchy;\nONS labels; definitions; hierarchy.',
 ('mesh_ons', 8, 'reasons'): '1. Food labelling gives information of food product.\n'
                             '2. Food product have food label.\n'
                             '3. Check the label before preparing a diet',
 ('mesh_ons', 8, 'evidence'): 'Food Labelling comment: Use of written, printed, or graphic materials upon or accompanying a food or its container or wrapper. '
                              'The concept includes ingredients, NUTRITIONAL VALUE, directions, warnings, and other relevant information.\n'
                              'Food material for humans and animals which is processed with the intention that it be consumable as a whole or added to other '
                              'food products.',
 ('mesh_ons', 8, 'evidence_fields'): 'MeSH definition; hierarchy;\nONS labels; definitions; hierarchy.',
 ('mesh_ons', 9, 'reasons'): '1.\n2.\n3.',
 ('mesh_ons', 9, 'evidence_fields'): 'MeSH definition; hierarchy;\nONS labels; definitions; hierarchy.',
 ('mesh_ons', 10, 'property_labels_raw'): 'processed_by,\nproduced_by,\nbasic_resource_for_functionality,\nused_by,\nsupplied_by,',
 ('mesh_ons', 10, 'reasons'): '1. Food Industry process food material\n2.\n3.',
 ('mesh_ons', 10, 'evidence'): 'Food Industry Comment: The industry concerned with processing, preparing, preserving, distributing, and serving of foods and '
                               'beverages.\n'
                               'Food material comment: for humans and animals which is processed with the intention that it be consumable as a whole or added '
                               'to other food products.',
 ('mesh_ons', 10, 'evidence_fields'): 'MeSH definition; hierarchy;\nONS labels; definitions; hierarchy.',
 ('mesh_ons', 11, 'property_labels_raw'): 'processed_by,\nincludes,\nproduced/distributes',
 ('mesh_ons', 11, 'reasons'): '1. Food Industry process food products.\n2.\n3.',
 ('mesh_ons', 11, 'evidence_fields'): 'MeSH definition; hierarchy;\nONS labels; definitions; hierarchy.',
 ('mesh_ons', 12, 'property_labels_raw'): 'processed_by,\nproduced_by',
 ('mesh_ons', 13, 'evidence_fields'): 'MeSH definition; hierarchy;\nONS definitions; hierarchy.',
 ('occo_mesh', 3, 'right_group'): 'Food preparation or serving related occupation holder\n'
                                  'chef or head cook;\n'
                                  'cook;\n'
                                  'cook, institution or cafeteria;\n'
                                  'cook, restaurant;\n'
                                  'bartender;\n'
                                  'waiter or waitress;\n'
                                  'food or beverage serving worker;',
 ('occo_mesh', 3, 'property_labels_raw'): 'participates_in,\nworks_in,\nhas_occupation_in,\ninvolved_in',
 ('occo_mesh', 3, 'reasons'): '1. food preparation or serving related occupation holder works in food industry (Pascal & Tarek)\n'
                              '2. These occupations, including chefs, cooks, bartenders, and waitstaff, make up the active workforce employed within the food '
                              'and beverage industry sector. (Suresh)',
 ('occo_mesh', 3, 'evidence'): 'MeSH Food Industry is defined as the industry concerned with processing, preparing, preserving, distributing, and serving '
                               'foods and beverages.\n'
                               'OccO concepts represent food-preparation and serving occupations.',
 ('occo_mesh', 3, 'evidence_fields'): 'MeSH definition; hierarchy;\nOccO labels; definitions; hierarchy.',
 ('occo_mesh', 4, 'right_group'): 'cook, institution or cafeteria;\nCook, restaurant;\nquality control analysis;',
 ('occo_mesh', 4, 'property_labels_raw'): 'assessed_by,\nchecked_by,\nmaintain_by,\nresponsible_for,\nrealize_by,\nevaluates_by,\nensure_by,\nmonitor_by',
 ('occo_mesh', 4, 'reasons'): '1. Chef will check this details before preparing food. (Vishva)\n'
                              '2. cook, institution or cafeteria;  Cook, restaurant;  responsible for food, therefore needs to be able to see/taste/smell food '
                              'quality. (Pascal)\n'
                              '3. OccO classes contains the skill quality control analysis, which concerns evaluating quality and performance. (Suresh)',
 ('occo_mesh', 4, 'evidence'): 'MeSH Food Quality: Ratings of the characteristics of food including flavor, appearance, nutritional content, and the amount of '
                               'microbial and chemical contamination..\n'
                               'OccO quality control analysis: A skill realized in conducting tests and inspections of products, services, or processes to '
                               'evaluate quality or performance.\n'
                               'The class cook, institution or cafeteria;\n'
                               'Cook, restaurant; contain this skill\n'
                               'quality control analysis, monitoring, problem sensitivity in their restrictions.',
 ('occo_mesh', 5, 'right_group'): 'cook, institution or cafeteria;\nCook, restaurant;\nquality control analysis;\nproblem sensitivity;',
 ('occo_mesh', 5, 'property_labels_raw'): 'check_by,\nmonitors_by,\nprevent_by,\ndetect_by\nManaged_by',
 ('occo_mesh', 5, 'reasons'): '1. Chef will check this details before preparing food.\n'
                              '2. Chef can detect food contamination.\n'
                              '3. will check details for cooking diet food in hospitals, …',
 ('occo_mesh', 5, 'evidence'): 'MeSH Food Contamination is the presence of harmful, unpalatable, or objectionable foreign substances in food before, during, '
                               'or after processing or storage.\n'
                               'OccO cook classes prepare food and include quality-control/problem-detection and monitoring skills.',
 ('occo_mesh', 6, 'right_group'): 'chef or head cook;\ncook, institution or cafeteria;\nCook, restaurant',
 ('occo_mesh', 6, 'property_labels_raw'): 'check_by,\nconsider_by,\ncontrol_by,\nmanaged_by',
 ('occo_mesh', 6, 'reasons'): '1. Chef will check this details before preparing food.\n2. will check details for cooking diet food in hospitals, ...',
 ('occo_mesh', 6, 'evidence'): 'MeSH Nutritive Value indicates the contribution of a food to the nutrient content of the diet and can be affected by handling, '
                               'storage, and processing.\n'
                               'OccO chef/cook classes prepare, season, cook, and sometimes plan menus or order supplies.',
 ('occo_mesh', 7, 'right_group'): 'reading comprehension;\nchef or head cook;\ncook, institution or cafeteria;\nCook, restaurant',
 ('occo_mesh', 7, 'property_labels_raw'): 'check_by,\nread_by,\nunderstands_by,\nuses_food_label_information',
 ('occo_mesh', 7, 'reasons'): '1. Chef will check this details before preparing food.\n'
                              '2. will check details for cooking diet food in hospitals, …\n'
                              '3. understand content of label for food preparation planning (i.e. recipes for different diets)',
 ('occo_mesh', 7, 'evidence'): 'MeSH Food Labeling includes written, printed, or graphic information on food containers, including ingredients, nutritional '
                               'value, directions, and warnings.\n'
                               'OccO class has reading comprehension skill realized in understanding written sentences and paragraphs in work related '
                               'documents. → suggesting the\n'
                               'ability to understand written food-related information.',
 ('occo_ons', 3, 'left_group'): 'Management +\nFood preparation role\nchef or head cook;\ncook;\ncook, institution or cafeteria;\ncook, restaurant;',
 ('occo_ons', 3, 'right_group'): 'diet by nutritonal composition;\n'
                                 'DASH diet;\n'
                                 'gluten free diet:\n'
                                 'ketogenic diet;\n'
                                 'low-carbohydrate, high-protien, high fat diet rural diet',
 ('occo_ons', 3, 'property_labels_raw'): 'checked_for,\nNeeds to know,\nImplemented,\nPrepares',
 ('occo_ons', 3, 'reasons'): '1. cooks need to know diets, servers in for example hospitals need to double check if patients get correct food according to '
                             'their diet.\n'
                             '2. Chef will  prepare the diet',
 ('occo_ons', 3, 'evidence'): 'ONS diet is a collection of food items with nutritional, safety, organoleptic, and social characteristics. Chef/head cook may '
                              'plan menu items and prepare food. Or cook, institution or cafeteria; cook, restaurant; prepare food',
 ('occo_ons', 3, 'evidence_fields'): 'OccO definition; hierarchy;\nONS labels; definitions; hierarchy.',
 ('occo_ons', 4, 'left_group'): 'Management +\nFood preparation role\nchef or head cook;\ncook;\ncook, institution or cafeteria;\ncook, restaurant;',
 ('occo_ons', 4, 'property_labels_raw'): 'Prepare,\nCreate,\nHandles,\ncooks',
 ('occo_ons', 4, 'reasons'): '1. The cook creates/cooks/prepares/realizes the "meal" (or food product).\n'
                             '2.  chef or head cook;  cook;  cook, institution or cafeteria;  cook, restaurant; prepares food products',
 ('occo_ons', 4, 'evidence_fields'): 'MeSH definition; hierarchy;\nONS labels; definitions; hierarchy.',
 ('occo_ons', 5, 'left_group'): 'Management +\nFood preparation role\nchef or head cook;\ncook;\ncook, institution or cafeteria;\ncook, restaurant;',
 ('occo_ons', 6, 'left_group'): 'Management +\nFood preparation role\nchef or head cook;\ncook;\ncook, institution or cafeteria;\ncook, restaurant;',
 ('occo_ons', 7, 'right_group'): 'Food product\nAnimal food product,\nanimal seafood product,\nfish food product,\nfungus food product,\nplant food product,',
 ('occo_ons', 7, 'evidence_fields'): 'MeSH definition; hierarchy;\nONS labels; definitions; hierarchy.'}


def _read_grouped_ground_truth(root: Path) -> pd.DataFrame:
    specs=[
        ("mesh_ons","MeSH_ONS.ods","MeSH concept","ONS concepts"),
        ("occo_mesh","MeSH_OccO.ods","MeSH concept","OccO concepts"),
        ("occo_ons","OccO_ONS.ods","OccO concept","ONS concepts"),
    ]
    out=[]
    for pair, fn, left, right in specs:
        path=root/"data/ground_truth"/fn
        df=pd.read_excel(path, engine="odf")
        for idx,r in df.iterrows():
            if pd.isna(r.get(left)) or pd.isna(r.get(right)):
                continue
            rownum=idx+2
            def value(field, source_col):
                v=r.get(source_col)
                if pd.isna(v): return np.nan
                base=str(v).strip()
                return GROUPED_TEXT_OVERRIDES.get((pair,rownum,field),base)
            out.append({
                "ontology_pair":pair,"ontology_pair_label":PAIR_LABELS[pair],
                "source_file":fn,"source_row":rownum,
                "left_group":value("left_group",left),"right_group":value("right_group",right),
                "direction_raw":value("direction_raw","Direction"),
                "property_labels_raw":value("property_labels_raw","Possible property labels"),
                "reviewer_agreement":r.get("No. of people Agree"),
                "reasons":value("reasons","Reasons"),"evidence":value("evidence","Evidence"),
                "evidence_fields":value("evidence_fields","Evidence fields used"),
            })
    return pd.DataFrame(out)


# Manual, transparent normalisation of the grouped positive reference.  These rules are
# intentionally explicit because the ODS files contain group captions, spelling variants,
# slash-separated shorthands, and one direction label referring to the wrong ontology.
REF_GROUP_SPECS = [
    # MeSH–ONS
    ("mesh_ons","MO_04",["Food Contamination"],["animal food product","animal seafood product","fish food product","food product","fungus food product","plant food product"],"MeSH→ONS","affects | risk | occurs_in | monitor_for","The ODS cell says MeSH→OccO inside the MeSH–ONS sheet; corrected to the participating ontologies."),
    ("mesh_ons","MO_08_10",["Food Industry"],["food material","food product"],"MeSH→ONS","processed_by | produced_by | supplied_by",None),
    ("mesh_ons","MO_06",["Food Labeling"],["food material","food product"],"MeSH→ONS","applies_to | gives_information | according_to","Food Labelling normalised to the ontology label Food Labeling."),
    ("mesh_ons","MO_05",["Food Quality"],["food material","food product"],"MeSH→ONS","has_quality | assessed_by | evaluates",None),
    ("mesh_ons","MO_07_10",["Food Technology"],["food material","food product"],"MeSH→ONS","processed_by | preserved_using | produced_by",None),
    ("mesh_ons","MO_02_03",["Nutritive Value"],["animal food product","animal seafood product","fish food product","food material","food product","fungus food product","plant food product"],"ONS→MeSH / MeSH→ONS","has | derives | defines | assigned_to",None),
    ("mesh_ons","MO_01",["Nutritive Value"],["DASH diet","diet by nutritional composition","gluten free diet","ketogenic diet","low-carbohydrate, high-protien, high fat diet","rural diet"],"ONS→MeSH","checked_for | has | characterized_by | defines",None),
    ("mesh_ons","MO_11",["Public Health"],["diet"],"MeSH→ONS","improves | contributes_to","Public Heath normalised to Public Health."),
    # OccO–MeSH (stored pair orientation OccO -> MeSH, independent of bridge direction)
    ("occo_mesh","OM_01",["bartender","chef or head cook","cook","cook, institution or cafeteria","cook, restaurant","food or beverage serving worker","food preparation or serving related occupation holder","waiter or waitress"],["Food Industry"],"OccO→MeSH","participates_in | works_in | involved_in",None),
    ("occo_mesh","OM_05",["chef or head cook","cook, institution or cafeteria","cook, restaurant","reading comprehension"],["Food Labeling"],"MeSH→OccO","read_by | understood_by | uses_food_label_information",None),
    ("occo_mesh","OM_04",["chef or head cook","cook, institution or cafeteria","cook, restaurant"],["Nutritive Value"],"MeSH→OccO","checked_by | considered_by | managed_by",None),
    ("occo_mesh","OM_03",["cook, institution or cafeteria","cook, restaurant","problem sensitivity","quality control analysis"],["Food Contamination"],"MeSH→OccO","checked_by | monitored_by | prevented_by | detected_by",None),
    ("occo_mesh","OM_02",["cook, institution or cafeteria","cook, restaurant","quality control analysis"],["Food Quality"],"MeSH→OccO","assessed_by | checked_by | ensures",None),
    # OccO–ONS
    ("occo_ons","OO_01",["chef or head cook","cook","cook, institution or cafeteria","cook, restaurant"],["DASH diet","diet by nutritional composition","gluten free diet","ketogenic diet","low-carbohydrate, high-protien, high fat diet","rural diet"],"OccO→ONS","checked_for | needs_to_know | implemented | prepares","Group caption excluded; four explicitly listed occupation classes retained."),
    ("occo_ons","OO_03",["chef or head cook","cook","cook, institution or cafeteria","cook, restaurant"],["food material"],"OccO→ONS","uses",None),
    ("occo_ons","OO_02",["chef or head cook","cook","cook, institution or cafeteria","cook, restaurant"],["food product"],"OccO→ONS","prepares | creates | handles | cooks",None),
    ("occo_ons","OO_05",["food or beverage serving worker"],["animal food product","animal seafood product","fish food product","food product","fungus food product","plant food product"],"OccO→ONS","served","food/beverages serving worker normalised to the ontology label."),
]


def _reference_tables(by_label):
    rows=[]
    for pair,gid,sources,targets,direction,props,note in REF_GROUP_SPECS:
        so,to=PAIR_ONTOLOGIES[pair]
        for s in sources:
            for t in targets:
                sr=by_label[(so,norm_label(s))]; tr=by_label[(to,norm_label(t))]
                rows.append({
                    "ontology_pair":pair,"source_ontology":so,"source_label":s,
                    "target_ontology":to,"target_label":t,"group_id":gid,
                    "bridge_direction":direction,"accepted_properties":props,
                    "reference_type":"exact","normalisation_note":note,
                    "source_label_norm":norm_label(s),"target_label_norm":norm_label(t),
                    "source_iri":sr["iri"],"target_iri":tr["iri"],
                })
    exact=pd.DataFrame(rows)
    pair_rank={p:i for i,p in enumerate(PAIR_ORDER)}
    exact['_pair_rank']=exact['ontology_pair'].map(pair_rank)
    exact=exact.sort_values(['_pair_rank','source_label_norm','target_label_norm'], kind='stable').drop(columns='_pair_rank').reset_index(drop=True)
    # sensitivity-only expansion of the grouped phrase "All food products"
    expanded=exact.copy()
    product_labels=["animal food product","animal seafood product","fish food product","fungus food product","plant food product"]
    extra=[]
    sources=["chef or head cook","cook","cook, institution or cafeteria","cook, restaurant"]
    for s in sources:
        for t in product_labels:
            sr=by_label[("occo",norm_label(s))]; tr=by_label[("ons",norm_label(t))]
            extra.append({
                "ontology_pair":"occo_ons","source_ontology":"occo","source_label":s,
                "target_ontology":"ons","target_label":t,"group_id":"OO_04",
                "bridge_direction":"OccO→ONS","accepted_properties":"prepares | cooks",
                "reference_type":"hierarchy_expanded",
                "normalisation_note":"Sensitivity analysis: 'All food products' expanded to the six explicitly represented product classes.",
                "source_label_norm":norm_label(s),"target_label_norm":norm_label(t),
                "source_iri":sr['iri'],"target_iri":tr['iri'],
            })
    expanded=pd.concat([expanded,pd.DataFrame(extra)],ignore_index=True)
    return exact, expanded


def _ground_truth_audit(grouped, exact, expanded):
    rows=[]
    for pair in PAIR_ORDER:
        g=grouped[grouped.ontology_pair==pair]
        anomaly=0
        if pair=="mesh_ons":
            anomaly=int(g.direction_raw.fillna('').str.contains('OccO',case=False).sum())
        rows.append({
            "ontology_pair":pair,"ontology_pair_label":PAIR_LABELS[pair],
            "grouped_rows":len(g),
            "rows_with_reviewer_agreement":int(g.reviewer_agreement.notna().sum()),
            "rows_missing_reviewer_agreement":int(g.reviewer_agreement.isna().sum()),
            "direction_label_anomalies":anomaly,
            "exact_pair_reference":int((exact.ontology_pair==pair).sum()),
            "expanded_pair_reference":int((expanded.ontology_pair==pair).sum()),
            "verified_negative_pairs":0,
            "primary_use":"positive pair-level support/retention",
            "unsupported_claims":"precision, accuracy, F1, exhaustive false-positive counts",
        })
    return pd.DataFrame(rows)


def _read_json_pairs(path: Path):
    try:
        data=json.load(open(path,encoding='utf-8'))
    except Exception:
        return set(),0
    if not isinstance(data,list): return set(),0
    pairs=set()
    for r in data:
        if isinstance(r,dict) and 'source' in r and 'target' in r:
            pairs.add((str(r['source']),str(r['target'])))
    return pairs,len(data)


def _stage1(root: Path, exact, expanded):
    all_rows=[]; manifests=[]; sets={}
    names={
        "Llama":"Llama-3.3-70B-Instruct_raw_s_f_{pair}_deeponto_stage1_case4",
        "Qwen":"Qwen3-32B_raw_s_f_{pair}_deeponto_stage1_case4",
    }
    for model in MODEL_ORDER:
        for pair in PAIR_ORDER:
            folder=root/f"data/stages/Stage_1/Stage1_{model}/{pair}"
            stem=names[model].format(pair=pair)
            csvp=folder/f"{stem}.csv"; jsonp=folder/f"{stem}.json"
            df=pd.read_csv(csvp).copy()
            df['source']=df['source'].astype(str);df['target']=df['target'].astype(str)
            df['pair_key']=[pair_key(a,b) for a,b in zip(df.source,df.target)]
            df['stage1_model']=model;df['ontology_pair']=pair;df['ontology_pair_label']=PAIR_LABELS[pair]
            pset=set(zip(df.source,df.target)); sets[(pair,model)]=pset
            jpairs,jcount=_read_json_pairs(jsonp)
            manifests.append({
                "stage1_model":model,"ontology_pair":pair,"ontology_pair_label":PAIR_LABELS[pair],
                "file":str(csvp.relative_to(root)),"json_file":str(jsonp.relative_to(root)),
                "rows":len(df),"unique_pairs":len(pset),"duplicate_rows":len(df)-len(pset),
                "missing_pair_keys":int(df.pair_key.eq('||').sum()),
                "csv_json_same_pairs":pset==jpairs,"csv_json_same_count":len(df)==jcount,
                "sha256_csv":sha256(csvp),"sha256_json":sha256(jsonp),
            })
            all_rows.append(df)
    stage1_rows=pd.concat(all_rows,ignore_index=True)
    manifest=pd.DataFrame(manifests)
    metrics=[]
    ontology_n={"mesh":13,"occo":52,"ons":26}
    for model in MODEL_ORDER:
        for pair in PAIR_ORDER:
            so,to=PAIR_ONTOLOGIES[pair]; s=sets[(pair,model)]
            er=exact[exact.ontology_pair==pair]
            xr=expanded[expanded.ontology_pair==pair]
            eset=set(zip(er.source_iri,er.target_iri)); xset=set(zip(xr.source_iri,xr.target_iri))
            metrics.append({
                "stage1_model":model,"ontology_pair":pair,"ontology_pair_label":PAIR_LABELS[pair],
                "pair_universe":ontology_n[so]*ontology_n[to],"candidate_pairs":len(s),
                "candidate_selection_rate":len(s)/(ontology_n[so]*ontology_n[to]),
                "exact_reference_pairs":len(eset),"exact_reference_retained":len(s&eset),
                "exact_reference_coverage":len(s&eset)/len(eset),
                "expanded_reference_pairs":len(xset),"expanded_reference_retained":len(s&xset),
                "expanded_reference_coverage":len(s&xset)/len(xset),
            })
    metrics=pd.DataFrame(metrics)
    overlap=[]
    for pair in PAIR_ORDER:
        a=sets[(pair,'Llama')]; b=sets[(pair,'Qwen')]
        overlap.append({"ontology_pair":pair,"ontology_pair_label":PAIR_LABELS[pair],
                        "llama_only":len(a-b),"shared":len(a&b),"qwen_only":len(b-a),
                        "union":len(a|b),"jaccard":len(a&b)/len(a|b)})
    return stage1_rows,manifest,pd.DataFrame(metrics),pd.DataFrame(overlap),sets


def _balanced_json_objects(raw):
    s='' if pd.isna(raw) else str(raw)
    out=[]; depth=0; start=None; ins=False; esc=False
    for i,ch in enumerate(s):
        if ins:
            if esc: esc=False
            elif ch=='\\': esc=True
            elif ch=='"': ins=False
            continue
        if ch=='"': ins=True; continue
        if ch=='{':
            if depth==0: start=i
            depth+=1
        elif ch=='}' and depth>0:
            depth-=1
            if depth==0 and start is not None:
                txt=s[start:i+1]
                try:
                    obj=json.loads(txt)
                    if isinstance(obj,dict): out.append((start,i+1,obj,txt))
                except Exception: pass
                start=None
    return out


def _audit_raw(raw):
    s='' if pd.isna(raw) else str(raw); stripped=s.strip()
    raw_obj=None
    try:
        o=json.loads(stripped)
        if isinstance(o,dict): raw_obj=o
    except Exception: pass
    objects=_balanced_json_objects(s)
    # if strict parsing succeeded, balanced scan will normally find same object; ensure it is represented
    if raw_obj is not None and not objects:
        p=s.find(stripped); objects=[(p,p+len(stripped),raw_obj,stripped)]
    raw_parseable=raw_obj is not None

    def dec(o):
        v=o.get('related') if isinstance(o,dict) else None
        if isinstance(v,str) and v.strip().lower() in {'yes','no'}: return v.strip().lower()
        return None
    decisions=[dec(o) for _,_,o,_ in objects if dec(o) is not None]
    uniq_dec=sorted(set(decisions))
    decision=uniq_dec[0] if len(uniq_dec)==1 else None
    decision_recoverable=decision is not None

    relationships=[]
    for _,_,o,_ in objects:
        if dec(o)==decision and 'relationship' in o:
            v=o.get('relationship')
            if v is None: v=''
            if isinstance(v,str): relationships.append(v.strip())
    nonempty=sorted(set(x for x in relationships if x!=''))
    # for a positive decision one unique non-empty relation is required; for no, empty/missing is sufficient
    if decision=='yes':
        relationship=nonempty[0] if len(nonempty)==1 else None
        record_recoverable=(relationship is not None)
    elif decision=='no':
        relationship=''
        record_recoverable=True
    else:
        relationship=None; record_recoverable=False

    # recovered schema conformity: an exact two-field object exists that agrees with recovered decision/relationship
    def schema_obj(o):
        return isinstance(o,dict) and set(o.keys())=={'related','relationship'} and dec(o) is not None and isinstance(o.get('relationship'),str)
    exact_objs=[o for _,_,o,_ in objects if schema_obj(o)]
    raw_schema_conformant = raw_obj is not None and schema_obj(raw_obj)
    # state consistency does not require the relationship key for no
    state_consistent = (decision=='yes' and record_recoverable and relationship!='') or (decision=='no' and decision_recoverable)
    schema_conformant=False
    if record_recoverable:
        for o in exact_objs:
            d=dec(o); rel=o.get('relationship','').strip()
            if d==decision and ((d=='yes' and rel==relationship and rel!='') or (d=='no' and rel=='')):
                schema_conformant=True; break

    # choose stored relationship even when decision conflicts if there is exactly one non-empty relation in all objects
    if relationship is None:
        allrels=[]
        for _,_,o,_ in objects:
            v=o.get('relationship') if isinstance(o,dict) else None
            if isinstance(v,str) and v.strip(): allrels.append(v.strip())
        ur=sorted(set(allrels))
        stored_relationship=ur[0] if len(ur)==1 else None
    else:
        stored_relationship=relationship if relationship!='' else None

    defects=[]
    if '```' in s: defects.append('code_fence')
    if re.search(r'(?is)###\s*(?:Concept\s*[12]\s*:|concept\s*:)', s): defects.append('prompt_echo')
    if not raw_parseable: defects.append('extra_text')
    if len(objects)>1: defects.append('repeated_json')
    if decision=='yes' and len(nonempty)>1: defects.append('ambiguous_relationship')
    if len(uniq_dec)>1: defects.append('conflicting_decision')
    if any('relationship' not in o for _,_,o,_ in objects): defects.append('missing_relationship')
    if any(set(o.keys())-{'related','relationship'} for _,_,o,_ in objects): defects.append('extra_keys')
    if not objects: defects.append('no_parseable_object')

    return {
        'raw_parseable':raw_parseable,'raw_schema_conformant':raw_schema_conformant,
        'decision_recoverable':decision_recoverable,'record_recoverable':record_recoverable,
        'schema_conformant':schema_conformant,'state_consistent':state_consistent,
        'decision':decision,'relationship':stored_relationship,'json_object_count':len(objects),
        'defects':' | '.join(defects) if defects else np.nan,
        'code_fence':'code_fence' in defects,'prompt_echo':'prompt_echo' in defects,'extra_text':'extra_text' in defects,
        'repeated_json':'repeated_json' in defects,'ambiguous_relationship':'ambiguous_relationship' in defects,
        'conflicting_decision':'conflicting_decision' in defects,'missing_relationship':'missing_relationship' in defects,
        'extra_keys':'extra_keys' in defects,'no_parseable_object':'no_parseable_object' in defects,
    }


def _run_id(pair, model, examples, direction):
    return f"{PAIR_PREFIX[pair]}_{'L' if model=='Llama' else 'Q'}_{'E0' if examples=='without' else 'E1'}_{'D12' if direction=='1-2' else 'D21'}"


def _stage2(root: Path, stage1_sets, exact, expanded):
    manifest=[]; allrows=[]; run_tables={}
    for pair in PAIR_ORDER:
        for model in MODEL_ORDER:
            expected=stage1_sets[(pair,model)]
            for ex in EXAMPLE_ORDER:
                for dr in DIRECTION_ORDER:
                    rid=_run_id(pair,model,ex,dr)
                    stem=f"{model}_Stage1_{STAGE2_MODEL}_raw_s_f_{pair}_stage2_case6_{'without_eg' if ex=='without' else 'with_eg'}_{dr}"
                    csvp=root/f"data/stages/Stage_2/{pair}/{stem}.csv"; jsonp=root/f"data/stages/Stage_2/{pair}/{stem}.json"
                    status='missing_file'; note='No corresponding CSV file is present.'; df=pd.DataFrame(); pset=set(); jpairs=set();jcount=0
                    if csvp.exists():
                        try: df=pd.read_csv(csvp)
                        except Exception: df=pd.DataFrame()
                        if len(df)==0:
                            status='empty_run'; note='The Stage 2 file is empty although Stage 1 retained candidates.'
                        else:
                            df=df.copy(); df['source']=df.source.astype(str);df['target']=df.target.astype(str)
                            pset=set(zip(df.source,df.target))
                            miss=expected-pset; unexp=pset-expected
                            if not miss and not unexp:
                                status='complete'; note=None
                            else:
                                other='Qwen' if model=='Llama' else 'Llama'
                                if pset==stage1_sets[(pair,other)]:
                                    status='candidate_provenance_mismatch'; note=f"Observed pair set equals the {other} Stage 1 candidate set exactly."
                                else:
                                    status='pair_set_mismatch'; note='Observed Stage 2 pair set does not match the declared Stage 1 candidate set.'
                    miss=len(expected-pset); unexp=len(pset-expected)
                    if jsonp.exists(): jpairs,jcount=_read_json_pairs(jsonp)
                    observed=len(df); uniq=len(pset); analysis_included=status=='complete'
                    manifest.append({
                        'run_id':rid,'stage1_model':model,'stage2_model':STAGE2_MODEL,'ontology_pair':pair,'ontology_pair_label':PAIR_LABELS[pair],
                        'examples':ex,'direction':dr,'file':str(csvp.relative_to(root)) if csvp.exists() else np.nan,
                        'expected_candidates':len(expected),'observed_rows':observed,'unique_pairs':uniq,
                        'duplicate_rows':(observed-uniq) if csvp.exists() else np.nan,'missing_expected_pairs':miss,'unexpected_pairs':unexp,
                        'run_coverage':(len(pset&expected)/len(expected)) if (csvp.exists() and len(expected)) else np.nan,
                        'unexpected_pair_rate':(unexp/uniq) if uniq else (0.0 if status=='empty_run' else np.nan),
                        'csv_json_same_pairs': (pset==jpairs) if (csvp.exists() and jsonp.exists()) else np.nan,
                        'csv_json_same_count': (observed==jcount) if (csvp.exists() and jsonp.exists()) else np.nan,
                        'status':status,'analysis_included':analysis_included,'provenance_note':note,
                        'sha256_csv':sha256(csvp) if csvp.exists() else np.nan,'sha256_json':sha256(jsonp) if jsonp.exists() else np.nan,
                    })
                    if observed:
                        audits=pd.DataFrame([_audit_raw(x) for x in df.raw_output])
                        full=pd.concat([df.reset_index(drop=True),audits],axis=1)
                        full['run_id']=rid;full['ontology_pair']=pair;full['ontology_pair_label']=PAIR_LABELS[pair]
                        full['stage1_model']=model;full['stage2_model']=STAGE2_MODEL;full['examples']=ex;full['direction']=dr
                        full['pair_key']=[pair_key(a,b) for a,b in zip(full.source,full.target)]
                        full['analysis_included']=analysis_included
                        run_tables[rid]=full
                        allrows.append(full)
    manifest=pd.DataFrame(manifest)
    stage2_rows=pd.concat(allrows,ignore_index=True) if allrows else pd.DataFrame()

    # metrics only for provenance-consistent complete runs
    exact_sets={p:set(zip(exact.loc[exact.ontology_pair==p,'source_iri'],exact.loc[exact.ontology_pair==p,'target_iri'])) for p in PAIR_ORDER}
    expanded_sets={p:set(zip(expanded.loc[expanded.ontology_pair==p,'source_iri'],expanded.loc[expanded.ontology_pair==p,'target_iri'])) for p in PAIR_ORDER}
    metrics=[]
    defect_names=['code_fence','prompt_echo','extra_text','repeated_json','ambiguous_relationship','conflicting_decision','missing_relationship','extra_keys','no_parseable_object']
    for _,mr in manifest[manifest.analysis_included].iterrows():
        rid=mr.run_id; df=run_tables[rid]; n=len(df); expected=stage1_sets[(mr.ontology_pair,mr.stage1_model)]
        dvalid=df.decision_recoverable
        yes=df.decision.eq('yes') & dvalid
        rec={
            'run_id':rid,'stage1_model':mr.stage1_model,'stage2_model':STAGE2_MODEL,'ontology_pair':mr.ontology_pair,'ontology_pair_label':mr.ontology_pair_label,
            'examples':mr.examples,'direction':mr.direction,'expected_candidates':len(expected),'observed_rows':n,'run_coverage':mr.run_coverage,
            'raw_parseability':df.raw_parseable.mean(),'raw_schema_conformity':df.raw_schema_conformant.mean(),
            'decision_recoverability':df.decision_recoverable.mean(),'record_recoverability':df.record_recoverable.mean(),
            'schema_conformity':df.schema_conformant.mean(),'state_consistency':df.state_consistent.mean(),
            'repair_rate':(~df.raw_parseable).mean(),
            'decision_recovery_gain':df.decision_recoverable.mean()-df.raw_parseable.mean(),
            'record_recovery_gain':df.record_recoverable.mean()-df.raw_parseable.mean(),
            'decision_usable_n':int(df.decision_recoverable.sum()),'ambiguous_decision_n':int((~df.decision_recoverable).sum()),
            'yes_n':int(yes.sum()),'no_n':int((df.decision.eq('no')&dvalid).sum()),
            'positive_rate':yes.sum()/max(df.decision_recoverable.sum(),1),
            'confirmation_rate':yes.sum()/len(expected),
        }
        yset=set(zip(df.loc[yes,'source'],df.loc[yes,'target']))
        for scope,rsets,prefix in [('exact',exact_sets,'exact'),('expanded',expanded_sets,'expanded')]:
            R=rsets[mr.ontology_pair]; denom=len(R&expected); hit=len(R&yset)
            rec[f'{prefix}_reference_in_stage1']=denom
            rec[f'{prefix}_reference_yes']=hit
            rec[f'{prefix}_stage2_conditional_retention']=hit/denom if denom else np.nan
            rec[f'{prefix}_end_to_end_reference_support']=hit/len(R) if R else np.nan
        for d in defect_names:
            cnt=int(df[d].sum())
            rec[f'defect_{d}_rate']=cnt/n
            rec[f'defect_{d}_n']=cnt
        metrics.append(rec)
    metrics=pd.DataFrame(metrics)
    return manifest,stage2_rows,metrics,run_tables


def _reference_fates(manifest, run_tables, stage1_sets, reference, scope):
    out=[]
    for _,mr in manifest[manifest.analysis_included].iterrows():
        R=reference[reference.ontology_pair==mr.ontology_pair]
        S=stage1_sets[(mr.ontology_pair,mr.stage1_model)]
        df=run_tables[mr.run_id]
        dmap={ (str(r.source),str(r.target)): r.decision for _,r in df[df.decision_recoverable].iterrows() }
        for _,r in R.iterrows():
            k=(str(r.source_iri),str(r.target_iri))
            if k not in S: fate='lost_in_stage1'
            elif k not in dmap: fate='stage2_unavailable_or_ambiguous'
            elif dmap[k]=='yes': fate='stage2_yes'
            else: fate='stage2_no'
            out.append({'reference_scope':scope,'run_id':mr.run_id,'ontology_pair':mr.ontology_pair,'ontology_pair_label':mr.ontology_pair_label,
                        'stage1_model':mr.stage1_model,'examples':mr.examples,'direction':mr.direction,
                        'pair_key':pair_key(*k),'source_label':r.source_label,'target_label':r.target_label,'fate':fate})
    return pd.DataFrame(out)


def _kappa(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if len(a)==0:return np.nan
    po=np.mean(a==b); pa=np.mean(a=='yes'); pb=np.mean(b=='yes'); pe=pa*pb+(1-pa)*(1-pb)
    return (po-pe)/(1-pe) if pe<1 else np.nan


def _compare_runs(run_tables, ra, rb, ctype, contrast, pair, s1):
    A=run_tables[ra]
    B=run_tables[rb]
    A=A[A.decision_recoverable][['source','target','decision']].rename(columns={'decision':'a'})
    B=B[B.decision_recoverable][['source','target','decision']].rename(columns={'decision':'b'})
    M=A.merge(B,on=['source','target'])
    tab=pd.crosstab(M.a,M.b).reindex(index=['yes','no'],columns=['yes','no'],fill_value=0)
    ag=float((M.a==M.b).mean()) if len(M) else np.nan
    return {'comparison_type':ctype,'contrast':contrast,'run_a':ra,'run_b':rb,'ontology_pair':pair,'ontology_pair_label':PAIR_LABELS[pair],
            'stage1_model':s1,'comparable_pairs':len(M),'agreement':ag,'decision_change_rate':1-ag if len(M) else np.nan,
            'cohen_kappa':_kappa(M.a.values,M.b.values),'yes_yes':int(tab.loc['yes','yes']),'yes_no':int(tab.loc['yes','no']),
            'no_yes':int(tab.loc['no','yes']),'no_no':int(tab.loc['no','no'])}


def _comparisons(manifest, run_tables):
    inc=manifest[manifest.analysis_included].copy(); out=[]
    # per ontology pair and selector: direction and example contrasts
    for pair in PAIR_ORDER:
        for model in MODEL_ORDER:
            g=inc[(inc.ontology_pair==pair)&(inc.stage1_model==model)]
            for ex in EXAMPLE_ORDER:
                ids={r.direction:r.run_id for _,r in g[g.examples==ex].iterrows()}
                if '1-2' in ids and '2-1' in ids:
                    out.append(_compare_runs(run_tables,ids['1-2'],ids['2-1'],'direction',f"{'without' if ex=='without' else 'with'} examples: 1→2 vs 2→1",pair,model))
            for dr in DIRECTION_ORDER:
                ids={r.examples:r.run_id for _,r in g[g.direction==dr].iterrows()}
                if 'without' in ids and 'with' in ids:
                    out.append(_compare_runs(run_tables,ids['without'],ids['with'],'examples',f"direction {dr}: without vs with examples",pair,model))
        # compare the two Stage-1 selectors downstream only on common pairs
        for ex in EXAMPLE_ORDER:
            for dr in DIRECTION_ORDER:
                a=inc[(inc.ontology_pair==pair)&(inc.stage1_model=='Llama')&(inc.examples==ex)&(inc.direction==dr)]
                b=inc[(inc.ontology_pair==pair)&(inc.stage1_model=='Qwen')&(inc.examples==ex)&(inc.direction==dr)]
                if len(a) and len(b):
                    out.append(_compare_runs(run_tables,a.iloc[0].run_id,b.iloc[0].run_id,'stage1_selector',f"{'without' if ex=='without' else 'with'} examples, direction {dr}: Llama-S1 vs Qwen-S1",pair,'Llama vs Qwen'))
    return pd.DataFrame(out)


def _direction_queue(manifest,run_tables):
    out=[]
    for _,mr in manifest[manifest.analysis_included].iterrows():
        df=run_tables[mr.run_id]
        for _,r in df[(df.decision=='yes') & df.decision_recoverable].iterrows():
            expected_subject=r.source_label if mr.direction=='1-2' else r.target_label
            expected_object=r.target_label if mr.direction=='1-2' else r.source_label
            rel=np.nan if pd.isna(r.relationship) else str(r.relationship)
            ns=norm_label(expected_subject); no=norm_label(expected_object); nr=norm_label(rel) if not pd.isna(rel) else ''
            ps=nr.find(ns) if ns else -1; po=nr.find(no) if no else -1
            if ps>=0 and po>=0:
                tri='expected_subject_before_object' if ps<po else 'expected_object_before_subject'
                priority='standard_review' if ps<po else 'priority_review'
            else:
                tri='one_or_both_labels_not_literal';priority='priority_review'
            out.append({'run_id':mr.run_id,'ontology_pair':mr.ontology_pair,'ontology_pair_label':mr.ontology_pair_label,'stage1_model':mr.stage1_model,
                        'examples':mr.examples,'direction':mr.direction,'source_iri':r.source,'source_label':r.source_label,'target_iri':r.target,'target_label':r.target_label,
                        'expected_subject':expected_subject,'expected_object':expected_object,'relationship':rel,'surface_triage':tri,'review_priority':priority,
                        'manual_direction_label':np.nan,'manual_direction_note':np.nan})
    return pd.DataFrame(out)


def _identity_audit(ontologies, stage1_rows, stage2_rows):
    iri_to_label=dict(zip(ontologies.iri,ontologies.label))
    rows=[]
    for stage,df in [('Stage 1',stage1_rows),('Stage 2',stage2_rows)]:
        if len(df)==0: continue
        src_known=df.source.astype(str).isin(iri_to_label)
        tgt_known=df.target.astype(str).isin(iri_to_label)
        src_match=[norm_label(lbl)==norm_label(iri_to_label.get(str(iri),'')) for iri,lbl in zip(df.source,df.source_label)]
        tgt_match=[norm_label(lbl)==norm_label(iri_to_label.get(str(iri),'')) for iri,lbl in zip(df.target,df.target_label)]
        rows.append({'stage':stage,'rows':len(df),'source_iri_known_rate':src_known.mean(),'target_iri_known_rate':tgt_known.mean(),
                     'source_label_match_rate':np.mean(src_match),'target_label_match_rate':np.mean(tgt_match)})
    return pd.DataFrame(rows)


def _metric_profile(stage2_metrics):
    metrics=['run_coverage','raw_parseability','decision_recoverability','record_recoverability','schema_conformity','repair_rate','positive_rate','confirmation_rate','exact_stage2_conditional_retention','exact_end_to_end_reference_support']
    rows=[]; total=len(stage2_metrics)
    for m in metrics:
        s=pd.to_numeric(stage2_metrics[m],errors='coerce').dropna()
        rows.append({'metric':m,'applicable_runs':len(s),'applicability_rate':len(s)/total if total else np.nan,
                     'minimum':s.min() if len(s) else np.nan,'maximum':s.max() if len(s) else np.nan,
                     'range':s.max()-s.min() if len(s) else np.nan,'iqr':s.quantile(.75)-s.quantile(.25) if len(s) else np.nan,
                     'unique_values':s.nunique()})
    return pd.DataFrame(rows)


def build_analysis(project_root: Path | str):
    root=Path(project_root)
    ontologies,evidence,iri_lookup,by_label=_load_ontologies(root)
    grouped=_read_grouped_ground_truth(root)
    exact,expanded=_reference_tables(by_label)
    gt_audit=_ground_truth_audit(grouped,exact,expanded)
    stage1_rows,stage1_manifest,stage1_metrics,overlap,stage1_sets=_stage1(root,exact,expanded)
    stage2_manifest,stage2_rows,stage2_metrics,run_tables=_stage2(root,stage1_sets,exact,expanded)
    exact_fates=_reference_fates(stage2_manifest,run_tables,stage1_sets,exact,'exact')
    expanded_fates=_reference_fates(stage2_manifest,run_tables,stage1_sets,expanded,'expanded')
    comparisons=_comparisons(stage2_manifest,run_tables)
    direction_queue=_direction_queue(stage2_manifest,run_tables)
    identity=_identity_audit(ontologies,stage1_rows,stage2_rows)
    metric_profile=_metric_profile(stage2_metrics)
    return {
        'ontologies':ontologies,'evidence_profile':evidence,'identity_audit':identity,
        'grouped_ground_truth':grouped,'ground_truth_audit':gt_audit,'exact_reference':exact,'expanded_reference':expanded,
        'stage1_rows':stage1_rows,'stage1_manifest':stage1_manifest,'stage1_metrics':stage1_metrics,'candidate_overlap':overlap,
        'stage2_manifest':stage2_manifest,'stage2_rows':stage2_rows,'stage2_metrics':stage2_metrics,
        'exact_fates':exact_fates,'expanded_fates':expanded_fates,'comparisons':comparisons,
        'direction_review_queue':direction_queue,'metric_profile':metric_profile,
    }
