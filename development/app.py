from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import re
import string
from collections import defaultdict
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import RegexpTokenizer
from fuzzywuzzy import process, fuzz
import os 

app = Flask(__name__)

# --- Path ke model dan data ---
MODEL_PATH = "./saved_model" 
LAPTOP_DATA_PATH = "cleaned_dataset_v3.csv"
MIN_REQ_DATA_PATH = "minimum_requirements_processed.csv"
REC_REQ_DATA_PATH = "recommended_requirements_processed.csv"

# --- Muat Model SentenceTransformer ---
try:
    print(f"Memuat model SentenceTransformer dari: {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH)
    print("Model SentenceTransformer berhasil dimuat.")
except Exception as e:
    print(f"Error memuat model SentenceTransformer: {e}")
    model = None 

# --- Muat Data dan Buat Embeddings/Knowledge Bases ---
laptop_df = None
min_req_df = None
rec_req_df = None
description_embeddings = None
min_req_kb = {}
rec_req_kb = {}
laptop_kb = defaultdict(list) 
category_kb = {} 
game_list = []
laptop_list = []
laptop_brand_list = []
unique_keyword_game_map = {}
game_series_map = {}

try:
    print("Memuat dataset...")
    laptop_df = pd.read_csv(LAPTOP_DATA_PATH)
    min_req_df = pd.read_csv(MIN_REQ_DATA_PATH)
    rec_req_df = pd.read_csv(REC_REQ_DATA_PATH)
    print("Dataset berhasil dimuat.")

    # --- Preprocessing dan Feature Engineering (dari notebook Anda) ---
    def clean_ram(ram_str):
        if pd.isna(ram_str) or str(ram_str).strip() == '' or str(ram_str).lower() == 'unknown':
            return 0
        ram_str = str(ram_str).lower().strip()
        if 'gb' in ram_str:
            return int(float(ram_str.replace('gb', '').strip()))
        elif 'mb' in ram_str:
            mb_value = float(ram_str.replace('mb', '').strip())
            return int(mb_value / 1024) if mb_value >= 512 else 1
        else:
            try:
                return int(float(ram_str))
            except ValueError:
                return 0

    def clean_file_size(size_str):
        if pd.isna(size_str) or str(size_str).strip() == '' or str(size_str).lower() == 'unknown':
            return 0
        size_str = str(size_str).lower().strip()
        if 'gb' in size_str:
            return int(float(size_str.replace('gb', '').strip()))
        elif 'mb' in size_str:
            mb_value = float(size_str.replace('mb', '').strip())
            return int(mb_value / 1024) if mb_value >= 512 else 1
        else:
            try:
                return int(float(size_str))
            except ValueError:
                return 0

    min_req_df['RAM'] = min_req_df['RAM'].apply(clean_ram)
    rec_req_df['RAM'] = rec_req_df['RAM'].apply(clean_ram)
    min_req_df['File Size'] = min_req_df['File Size'].apply(clean_file_size)
    rec_req_df['File Size'] = rec_req_df['File Size'].apply(clean_file_size)

    # Clean Description (Ensure clean_description function is here)
    def clean_description(desc):
        if pd.isnull(desc):
            return ""
        desc = re.sub(r'\n', ' ', desc)
        desc = re.sub(r'["\']', '', desc)
        desc = re.sub(r'[^\x00-\x7F]+', ' ', desc)
        desc = re.sub(r'\s+', ' ', desc)
        return desc.strip()

    rec_req_df['Description'] = rec_req_df['Description'].apply(clean_description)
    min_req_df['Description'] = min_req_df['Description'].apply(clean_description)


    # Create Knowledge Bases
    min_req_kb = min_req_df.set_index('Game').to_dict('index')
    rec_req_kb = rec_req_df.set_index('Game').to_dict('index')

    game_list = min_req_df['Game'].tolist()
    laptop_list = laptop_df['Model'].tolist()
    laptop_brand_list = laptop_df['Brand'].unique().tolist()

    # === NLP Helpers (Ensure these functions are in your Flask file) ===
    tokenizer = RegexpTokenizer(r'\w+')
    factory = StopWordRemoverFactory()
    stopwords_sastrawi = set(factory.get_stop_words())
    additional_stopwords = {
        'cocok', 'buat', 'main', 'dengan', 'rekomendasi', 'spesifikasi', 'spek',
        'apa', 'yang', 'laptop', 'harga', 'rp', 'ribu', 'juta', 'budget',
        'model', 'merek', 'brand', 'merk', 'duit', 'uang', 'dana', 'termurah'
    }
    stopwords_sastrawi.update(additional_stopwords)

    SYNONYM_MAPPING = {
        "aksi": "action", "laga": "action", "action": "action",
        "petualangan": "adventure", "openworld": "adventure", "open-world": "adventure", "adventure": "adventure",
        "animasi": "animation & modeling", "modeling": "animation & modeling", "animasi 3d": "animation & modeling",
        "santai": "casual", "kasual": "casual", "casual": "casual",
        "desain": "design & illustration", "ilustrasi": "design & illustration", "gambar": "design & illustration",
        "akses awal": "early access", "pre-release": "early access",
        "tembak": "fps", "tembak-tembakan": "fps", "first person": "fps", "fps": "fps",
        "pertarungan": "fighting", "fighter": "fighting", "bela diri": "fighting",
        "gratis": "free to play", "f2p": "free to play", "free": "free to play",
        "pengembangan game": "game development", "dev game": "game development", "pembuatan game": "game development",
        "kekerasan grafis": "gore", "darah": "gore", "gore": "gore",
        "indie": "indie", "independen": "indie", "permainan indie": "indie",
        "rpg jepang": "jrpg", "jrpg": "jrpg", "anime rpg": "jrpg",
        "mmo": "mmo", "multiplayer masif": "mmo", "game online": "mmo",
        "moba": "moba", "arena battle": "moba", "dota-like": "moba",
        "edit foto": "photo editing", "foto": "photo editing", "photoshop": "photo editing",
        "platform": "platformer", "lompat": "platformer", "platformer": "platformer",
        "role playing": "rpg", "rpg": "rpg", "peran": "rpg",
        "balapan": "racing", "balap": "racing", "racing": "racing",
        "roguelike": "rogue-like", "permadeath": "rogue-like", "prosedural": "rogue-like",
        "konten dewasa": "sexual content", "seksual": "sexual content", "dewasa": "sexual content",
        "simulasi": "simulation", "sim": "simulation", "simulator": "simulation",
        "angkasa": "space", "luar angkasa": "space", "space": "space",
        "olahraga": "sports", "sport": "sports", "olahraga elektronik": "sports",
        "strategi": "strategy", "strategic": "strategy", "taktik": "strategy",
        "tps": "third-person shooter", "shooter orang ketiga": "third-person shooter", "third person": "third-person shooter",
        "giliran": "turn-based", "turn based": "turn-based", "tb": "turn-based",
        "tidak diketahui": "unknown", "unknown": "unknown", "misteri": "unknown",
        "alat": "utilities", "tool": "utilities", "utilitas": "utilities",
        "produksi video": "video production", "editing video": "video production", "video editor": "video production",
        "kekerasan": "violent", "violent": "violent", "brutal": "violent"
    }

    def basic_preprocessing(text):
        text = str(text).lower()
        text = re.sub(r'[’‘]', "'", text)  # Normalisasi apostrof
        text = re.sub(f"[{string.punctuation}]", " ", text)
        tokens = tokenizer.tokenize(text)
        return tokens

    def remove_stopwords(tokens):
        return [t for t in tokens if t not in stopwords_sastrawi]

    def map_synonyms(tokens):
        return [SYNONYM_MAPPING.get(t, t) for t in tokens]

    def create_game_series_map(game_list):
        series_map = defaultdict(set)
        patterns_to_remove = [
            r'\s*\d+$',
            r'\s*\(?\d{4}\)?$',
            r'\s*(?:PES|HD|remastered|remake|definitive edition|enhanced edition|anniversary edition)\s*$',
            r'\s*:\s*.*$',
            r'\s*-+\s*.*$',
            r'\s*PES\s+\d{4}$',
        ]

        for game in game_list:
            series_name = game
            for pattern in patterns_to_remove:
                series_name = re.sub(pattern, '', series_name, flags=re.IGNORECASE).strip()
            series_name = re.sub(r'\s*PES\s*$', '', series_name, flags=re.IGNORECASE).strip()
            if series_name and series_name != game:
                series_map[series_name.lower()].add(game)
        return {series: list(games) for series, games in series_map.items() if len(games) > 1}

    game_series_map = create_game_series_map(game_list)

    def create_keyword_game_map(game_list, stopwords):
        keyword_candidate_map = defaultdict(list)
        for game in game_list:
            tokens = basic_preprocessing(game)
            tokens = remove_stopwords(tokens)
            for token in tokens:
                if len(token) > 2 or token.isdigit():
                    keyword_candidate_map[token].append(game)
        unique_keyword_map = {}
        for keyword, games in keyword_candidate_map.items():
            if len(games) == 1:
                unique_keyword_map[keyword] = games[0]
        return unique_keyword_map

    unique_keyword_game_map = create_keyword_game_map(game_list, stopwords_sastrawi)


    def handle_game_series(found_games, query_lower, game_series_map):
        found_games = set(found_games)
        series_to_add = set()
        def extract_number(game_name):
            numbers = re.findall(r'\d+', game_name)
            return int(numbers[-1]) if numbers else 0
        for series_name, games_in_series in game_series_map.items():
            if re.search(r'\b' + re.escape(series_name.lower()) + r'\b', query_lower):
                specific_found = any(game in found_games for game in games_in_series)
                if not specific_found and games_in_series:
                    sorted_games = sorted(games_in_series, key=extract_number, reverse=True)
                    series_to_add.add(sorted_games[0])
        found_games.update(series_to_add)
        return list(found_games)

    def extract_entities_and_budget(user_query, game_list, laptop_list, laptop_brand_list,
                                   unique_keyword_game_map, game_series_map,
                                   score_cutoff_high=95, score_cutoff_low_game=90,
                                   score_cutoff_low_laptop=80):
        found_games = set()
        found_laptops = set()
        extracted_budget = None
        query_lower = user_query.lower()
        query_tokens = basic_preprocessing(user_query)
        query_tokens_cleaned = remove_stopwords(query_tokens)
        budget_pattern = r'(?:harga|rp|rp\.|budget)\s*:?\s*(\d+(?:[,\.]\d+)?)\s*(rb|ribu|jt|juta)?|\b(\d+(?:[,\.]\d+)?)\s*(rb|ribu|jt|juta)\b'
        budget_match = re.search(budget_pattern, query_lower)
        budget_tokens = set()
        if budget_match:
            amount1 = budget_match.group(1)
            unit1 = budget_match.group(2)
            amount2 = budget_match.group(3)
            unit2 = budget_match.group(4)
            amount_str = amount1 if amount1 else amount2
            unit = unit1 if unit1 else unit2
            if amount_str:
                try:
                    amount = float(amount_str.replace(',', '.'))
                    if unit in ['juta', 'jt']: extracted_budget = int(amount * 1_000_000)
                    elif unit in ['ribu', 'rb']: extracted_budget = int(amount * 1_000)
                    else: extracted_budget = int(amount)
                    budget_tokens.add(amount_str.replace(',', '.'))
                    budget_tokens.add(amount_str.replace('.', ','))
                except ValueError: extracted_budget = None
        game_matching_tokens = [t for t in query_tokens_cleaned if t not in budget_tokens]
        cleaned_query_string_for_game = " ".join(game_matching_tokens)
        query_numbers = set(t for t in game_matching_tokens if t.isdigit())
        laptop_entities = set(laptop_list + laptop_brand_list)
        for entity in laptop_entities:
            if re.search(r'\b' + re.escape(entity.lower()) + r'\b', query_lower):
                found_laptops.add(entity)
        if not found_laptops:
            for entity in laptop_entities:
                match = process.extractOne(
                    entity, [query_lower], scorer=fuzz.token_set_ratio, score_cutoff=score_cutoff_high
                )
                if match: found_laptops.add(match[0]) # Corrected to add the matched entity
        laptop_tokens = set()
        for laptop in found_laptops:
            tokens = basic_preprocessing(laptop)
            tokens = remove_stopwords(tokens)
            laptop_tokens.update(tokens)
        game_matching_tokens = [
            t for t in query_tokens_cleaned if t not in budget_tokens and t not in laptop_tokens
        ]
        cleaned_query_string_for_game = " ".join(game_matching_tokens)
        query_numbers = set(t for t in game_matching_tokens if t.isdigit())
        matches_high = process.extractBests(
            query_lower, game_list, scorer=fuzz.token_set_ratio, score_cutoff=95, limit=5
        )
        for match, score in matches_high: found_games.add(match)
        for token in game_matching_tokens:
            if token in unique_keyword_game_map:
                game_from_keyword = unique_keyword_game_map[token]
                found_games.add(game_from_keyword)
        if not found_games:
            matches_low = process.extract( cleaned_query_string_for_game, game_list, limit=10 )
            for match, score in matches_low:
                if match in found_games: continue
                if score < score_cutoff_low_game: continue
                match_tokens = set(remove_stopwords(basic_preprocessing(match)))
                query_token_set = set(game_matching_tokens)
                overlap = match_tokens & query_token_set
                match_numbers = set(re.findall(r'\d+', match))
                if query_numbers and match_numbers:
                    if not (query_numbers & match_numbers): continue
                elif match_numbers and not query_numbers: continue
                min_required = max(1, min(2, len(match_tokens)//2))
                if len(overlap) >= min_required: found_games.add(match)
        found_games = handle_game_series(found_games, query_lower, game_series_map)
        if query_numbers:
            filtered_games = []
            for game in found_games:
                game_numbers = set(re.findall(r'\d+', game))
                if game_numbers:
                    if not (query_numbers & game_numbers): continue
                filtered_games.append(game)
            found_games = filtered_games
        laptop_entities = set(laptop_list + laptop_brand_list)
        for entity in laptop_entities:
            if re.search(r'\b' + re.escape(entity.lower()) + r'\b', query_lower):
                found_laptops.add(entity)
        if not found_laptops:
            for entity in laptop_entities:
                match = process.extractOne(
                    entity, [cleaned_query_string_for_game], scorer=fuzz.token_set_ratio, score_cutoff=score_cutoff_high
                )
                if match: found_laptops.add(match[0]) # Corrected to add the matched entity

        return list(found_games), list(found_laptops), extracted_budget

    def nlp_pipeline_fuzzy(user_query, game_list, laptop_list, laptop_brand_list,
                           unique_keyword_game_map, game_series_map):
        tokens = basic_preprocessing(user_query)
        tokens = remove_stopwords(tokens)
        tokens = map_synonyms(tokens)
        found_games, found_laptops, extracted_budget = extract_entities_and_budget(
            user_query, game_list, laptop_list, laptop_brand_list,
            unique_keyword_game_map, game_series_map,
            score_cutoff_low_game=90
        )
        return {
            "tokens": tokens,
            "found_games": found_games,
            "found_laptops": found_laptops,
            "budget": extracted_budget
        }

    def recognize_intent_simple(query, found_games, found_laptops, budget):
        query_lower = query.lower()
        query_tokens = set(basic_preprocessing(query_lower))
        intent = "FIND_LAPTOP_FOR_GAME"
        if "terbaik" in query_tokens or "tertinggi" in query_tokens or "high performance" in query_lower: intent = "FIND_BEST_LAPTOP_FOR_GAME"
        elif "termurah" in query_tokens or "paling murah" in query_lower or "low budget" in query_lower: intent = "FIND_CHEAPEST_LAPTOP_FOR_GAME"
        elif "bandingkan" in query_tokens or "compare" in query_tokens:
            if len(found_laptops) >= 2: intent = "COMPARE_LAPTOPS"
            else: intent = "UNKNOWN"
        elif (found_laptops or budget is not None) and not found_games: intent = "FILTER_LAPTOPS"
        if found_games and intent in ["FIND_LAPTOP_FOR_GAME", "FILTER_LAPTOPS"]: intent = "FIND_LAPTOP_FOR_GAME"
        if not found_games and not found_laptops and budget is None: intent = "GENERAL_QUERY"
        return intent

    def categorize_laptops_adapted(df_laptops_filtered, req_row_min, req_row_rec, ram_weight=1.0, storage_weight=1.0, storage_type_weight=0.5):
        if df_laptops_filtered.empty: return pd.DataFrame()
        ram_req = req_row_min.get('RAM', 0)
        storage_req = req_row_min.get('File Size', 0)

        def get_gpu_vendor(gpu_name):
            if pd.isna(gpu_name): return 'intel'
            gpu_name = str(gpu_name).lower()
            if any(x in gpu_name for x in ['rtx', 'gtx', 'mx']): return 'nvidia'
            elif any(x in gpu_name for x in ['radeon', 'rx', 'vega', 'pro']): return 'amd'
            elif any(x in gpu_name for x in ['integrated', 'iris', 'uhd', 'hd graphics']): return 'intel'
            else: return 'intel'

        def get_cpu_vendor(cpu_name):
            if pd.isna(cpu_name): return 'intel'
            cpu_name = str(cpu_name).lower()
            if 'intel' in cpu_name or any(x in cpu_name for x in ['core', 'pentium', 'celeron', 'xeon', 'evo']): return 'intel'
            elif 'amd' in cpu_name or any(x in cpu_name for x in ['ryzen', 'fx', 'athlon', 'phenom', 'opteron']): return 'amd'
            else: return 'intel'

        def get_cpu_req_score(row, req_row):
            vendor = get_cpu_vendor(row.get('CPU', '')) # Use .get with default
            if vendor == 'intel': return req_row.get('CPU_Intel_score', 0)
            elif vendor == 'amd': return req_row.get('CPU_AMD_score', 0)
            else: return req_row.get('CPU_Intel_score', 0)

        def get_gpu_req_score(row, req_row):
            vendor = get_gpu_vendor(row.get('GPU', '')) # Use .get with default
            if vendor == 'nvidia': return req_row.get('GPU_NVIDIA_score', 0)
            elif vendor == 'amd': return req_row.get('GPU_AMD_score', 0)
            elif vendor == 'intel': return req_row.get('GPU_Intel_score', 0)
            else: return req_row.get('GPU_Intel_score', 0)

        def get_storage_type_score(storage_type):
            if pd.isna(storage_type): return 0
            storage_type_lower = storage_type.lower()
            if 'ssd' in storage_type_lower: return 1.0
            elif 'hdd' in storage_type_lower: return 0.5
            elif 'emmc' in storage_type_lower: return 0.3
            else: return 0.0

        def calculate_match(row):
            cpu_req = get_cpu_req_score(row, req_row_rec)
            gpu_req = get_gpu_req_score(row, req_row_rec)

            cpu_score_ratio = row['CPU_score'] / cpu_req if cpu_req > 0 and not pd.isna(row['CPU_score']) else (1.0 if cpu_req == 0 else 0.0)
            gpu_score_ratio = row['GPU_score'] / gpu_req if gpu_req > 0 and not pd.isna(row['GPU_score']) else (1.0 if gpu_req == 0 else 0.0)
            ram_score_ratio = (row['RAM'] / ram_req) * ram_weight if ram_req > 0 and not pd.isna(row['RAM']) else (1.0 if ram_req == 0 else 0.0)
            storage_score_ratio = (row['Storage'] / storage_req) * storage_weight if storage_req > 0 and not pd.isna(row['Storage']) else (1.0 if storage_req == 0 else 0.0)
            storage_type_score = get_storage_type_score(row.get('Storage type', '')) * storage_type_weight # Use .get with default

            cpu_score_ratio = min(cpu_score_ratio, 5.0)
            gpu_score_ratio = min(gpu_score_ratio, 5.0)
            ram_score_ratio = min(ram_score_ratio, 5.0)
            storage_score_ratio = min(storage_score_ratio, 5.0)

            cpu_score_ratio = 0.0 if pd.isna(cpu_score_ratio) or cpu_score_ratio == float('inf') else cpu_score_ratio
            gpu_score_ratio = 0.0 if pd.isna(gpu_score_ratio) or gpu_score_ratio == float('inf') else gpu_score_ratio
            ram_score_ratio = 0.0 if pd.isna(ram_score_ratio) or ram_score_ratio == float('inf') else ram_score_ratio
            storage_score_ratio = 0.0 if pd.isna(storage_score_ratio) or storage_score_ratio == float('inf') else storage_score_ratio
            storage_type_score = 0.0 if pd.isna(storage_type_score) or storage_type_score == float('inf') else storage_type_score


            final_score = (cpu_score_ratio + gpu_score_ratio + ram_score_ratio + storage_score_ratio + storage_type_score) / (2 + ram_weight + storage_weight + storage_type_weight)
            return final_score

        def categorize(row):
            cpu_min = get_cpu_req_score(row, req_row_min)
            gpu_min = get_gpu_req_score(row, req_row_min)
            cpu_rec = get_cpu_req_score(row, req_row_rec)
            gpu_rec = get_gpu_req_score(row, req_row_rec)

            row_cpu_score = row.get('CPU_score', -1) if not pd.isna(row.get('CPU_score', -1)) else -1
            row_gpu_score = row.get('GPU_score', -1) if not pd.isna(row.get('GPU_score', -1)) else -1
            row_ram = row.get('RAM', -1) if not pd.isna(row.get('RAM', -1)) else -1
            row_storage = row.get('Storage', -1) if not pd.isna(row.get('Storage', -1)) else -1


            if row_cpu_score < cpu_min or row_gpu_score < gpu_min or row_ram < ram_req or row_storage < storage_req:
                 return 'Disqualified'

            cpu_rec_flag = row_cpu_score >= cpu_rec
            gpu_rec_flag = row_gpu_score >= gpu_rec

            if cpu_rec_flag and gpu_rec_flag: return 'Recommended'
            elif cpu_rec_flag or gpu_rec_flag: return 'Mixed'
            else: return 'Minimum'

        df_filtered_req = df_laptops_filtered[
            (df_laptops_filtered['RAM'] >= ram_req) & (df_laptops_filtered['Storage'] >= storage_req)
        ].copy()

        if df_filtered_req.empty: return pd.DataFrame()

        df_filtered_req['Match_Score'] = df_filtered_req.apply(calculate_match, axis=1)
        df_filtered_req['Category'] = df_filtered_req.apply(categorize, axis=1)

        df_final = df_filtered_req[df_filtered_req['Category'] != 'Disqualified'].copy()
        df_final = df_final.sort_values(by='Match_Score', ascending=False).reset_index(drop=True)
        return df_final[['Brand', 'Model', 'CPU', 'GPU', 'RAM', 'Storage', 'Storage type', 'Final Price', 'Category', 'Match_Score']]


    # === Semantic Search Function (Ensure this function is in your Flask file) ===
    def search_games(query, model, description_embeddings, df, top_n=5, category_kb=None, category_boost=1.0):
        if model is None:
             print("Model SentenceTransformer belum dimuat.")
             return []

        query_embedding = model.encode(query, convert_to_tensor=True)
        similarities = torch.nn.functional.cosine_similarity(query_embedding, description_embeddings)

        query_tokens_cleaned = basic_preprocessing(query)
        query_tokens_cleaned = remove_stopwords(query_tokens_cleaned)
        query_tokens_cleaned = map_synonyms(query_tokens_cleaned)

        boosted_similarities = similarities.clone()
        for i in range(len(df)):
            game_categories_str = df.iloc[i].get("Category")
            game_categories = []
            if isinstance(game_categories_str, str):
                 try: game_categories = eval(game_categories_str)
                 except: pass
            elif isinstance(game_categories_str, list): game_categories = game_categories_str
            if game_categories:
                game_categories_lower = set([cat.lower().strip() for cat in game_categories])
                if any(token in game_categories_lower for token in query_tokens_cleaned):
                    boosted_similarities[i] += category_boost

        top_indices = boosted_similarities.argsort(descending=True)[:top_n]
        results = []
        for idx in top_indices:
            idx = idx.item()
            results.append({
                "game": df.iloc[idx]["Game"],
                "category": df.iloc[idx].get("Category"), # Use .get for robustness
                "similarity": boosted_similarities[idx].item()
            })
        return results


    # === Main Recommendation Function (Ensure this function is in your Flask file) ===
    def get_laptop_recommendations_with_intent(user_query, laptop_df, min_req_df, rec_req_df,
                                   min_req_kb, rec_req_kb,
                                   model, description_embeddings,
                                   laptop_list, laptop_brand_list, unique_keyword_game_map, game_series_map,
                                   category_boost_value=1.0,
                                   top_n_semantic_games=10
                                   ):

        print(f"Query Pengguna: {user_query}")
        pipeline_result = nlp_pipeline_fuzzy(user_query, game_list, laptop_list, laptop_brand_list, unique_keyword_game_map, game_series_map)
        found_games = pipeline_result['found_games']
        found_laptops_entities = pipeline_result['found_laptops']
        extracted_budget = pipeline_result['budget']

        detected_intent = recognize_intent_simple(user_query, found_games, found_laptops_entities, extracted_budget)

        print(f"Hasil NLP Pipeline: Game={found_games}, Laptop/Entity={found_laptops_entities}, Budget={extracted_budget}")
        print(f"Intent Terdeteksi: {detected_intent}")

        if detected_intent == "COMPARE_LAPTOPS":
            if len(found_laptops_entities) < 2:
                 return pd.DataFrame({"Status": ["Informasi Kurang"], "Pesan": ["Mohon sebutkan minimal dua nama laptop atau brand untuk dibandingkan."]})
            print("\nIntent: Compare Laptops. Implementasi perbandingan spesifikasi laptop...")
            comparison_laptops = laptop_df[
                laptop_df['Model'].str.lower().isin([ent.lower() for ent in found_laptops_entities if ent in laptop_df['Model'].tolist()]) |
                laptop_df['Brand'].str.lower().isin([ent.lower() for ent in found_laptops_entities if ent in laptop_df['Brand'].unique().tolist()])
            ].copy()
            if comparison_laptops.empty:
                 return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": ["Tidak ada laptop yang cocok dengan yang ingin Anda bandingkan."]})
            return comparison_laptops[['Brand', 'Model', 'CPU', 'GPU', 'RAM', 'Storage', 'Storage type', 'Screen', 'Final Price']]

        elif detected_intent == "FILTER_LAPTOPS":
             if not found_laptops_entities and extracted_budget is None:
                  return pd.DataFrame({"Status": ["Informasi Kurang"], "Pesan": ["Mohon sebutkan brand laptop, model, atau budget yang Anda inginkan."]})
             print("\nIntent: Filter Laptops. Menerapkan filter berdasarkan kriteria yang terdeteksi...")
             laptops_to_evaluate = laptop_df.copy()
             if extracted_budget is not None and extracted_budget > 0:
                 print(f"  - Memfilter laptop dengan budget <= {extracted_budget}")
                 laptops_to_evaluate = laptops_to_evaluate[laptops_to_evaluate['Final Price'] <= extracted_budget].copy()
                 print(f"    Jumlah laptop setelah filter budget: {len(laptops_to_evaluate)}")
             if found_laptops_entities:
                 print(f"  - Memfilter laptop berdasarkan entitas: {found_laptops_entities}")
                 filter_mask = pd.Series([False] * len(laptops_to_evaluate), index=laptops_to_evaluate.index)
                 for entity in found_laptops_entities:
                     if entity in laptop_brand_list:
                         filter_mask |= (laptops_to_evaluate['Brand'].str.lower() == entity.lower())
                     elif entity in laptop_list:
                          filter_mask |= (laptops_to_evaluate['Model'].str.lower() == entity.lower())
                     else:
                         brand_match = process.extractOne(entity, laptops_to_evaluate['Brand'].unique().tolist(), score_cutoff=90)
                         model_match = process.extractOne(entity, laptops_to_evaluate['Model'].tolist(), score_cutoff=90)
                         if brand_match and brand_match[1] >= 90:
                              filter_mask |= (laptops_to_evaluate['Brand'].str.lower() == brand_match[0].lower())
                         if model_match and model_match[1] >= 90:
                              filter_mask |= (laptops_to_evaluate['Model'].str.lower() == model_match[0].lower())
                 laptops_to_evaluate = laptops_to_evaluate[filter_mask].copy()
                 print(f"    Jumlah laptop setelah filter entitas: {len(laptops_to_evaluate)}")

             if laptops_to_evaluate.empty:
                 print("Tidak ada laptop yang cocok dengan kriteria filter.")
                 return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": ["Tidak ada laptop yang cocok dengan kriteria filter Anda."]})
             return laptops_to_evaluate[['Brand', 'Model', 'CPU', 'GPU', 'RAM', 'Storage', 'Storage type', 'Final Price']]

        elif detected_intent in ["FIND_BEST_LAPTOP_FOR_GAME", "FIND_CHEAPEST_LAPTOP_FOR_GAME", "FIND_LAPTOP_FOR_GAME"]:
            game_targets = []
            if not found_games:
                print("\nTidak ada game spesifik terdeteksi dari query. Melakukan Semantic Search...")
                semantic_results = search_games(user_query, model, description_embeddings, min_req_df, top_n=top_n_semantic_games, category_boost=category_boost_value)
                game_targets = [res['game'] for res in semantic_results if res['similarity'] > 0.6]
                if game_targets:
                     print(f"  - Menemukan game relevan dari Semantic Search: {game_targets}")
                else:
                     print("  - Semantic Search tidak menemukan game relevan.")
                     return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": ["Tidak dapat menemukan game relevan dari query Anda."]})
            else:
                 game_targets = found_games
                 print(f"\nMenggunakan game yang terdeteksi spesifik oleh NLP Pipeline: {game_targets}")

            target_game_min_req = None
            target_game_rec_req = None
            target_game_name = None
            max_requirement_score = -1

            if game_targets:
                print(f"Mengevaluasi persyaratan untuk game target: {game_targets}")
                valid_game_targets = []
                for game_name in game_targets:
                    min_req = min_req_kb.get(game_name)
                    rec_req = rec_req_kb.get(game_name)
                    if min_req and rec_req:
                        valid_game_targets.append(game_name)
                        current_requirement_score = (min_req.get('CPU_Intel_score', 0) + min_req.get('CPU_AMD_score', 0) +
                                                     min_req.get('GPU_NVIDIA_score', 0) + min_req.get('GPU_AMD_score', 0) + min_req.get('GPU_Intel_score', 0))
                        print(f"  - {game_name}: Requirement Score = {current_requirement_score}")
                        if current_requirement_score > max_requirement_score:
                            max_requirement_score = current_requirement_score
                            target_game_name = game_name
                            target_game_min_req = min_req
                            target_game_rec_req = rec_req

                if not target_game_name:
                     print("Tidak dapat menemukan persyaratan yang valid untuk game target dari knowledge base.")
                     return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": ["Tidak dapat menemukan persyaratan yang valid untuk game target dari knowledge base."]})

                game_targets = valid_game_targets
                print(f"\nMemilih game dengan persyaratan tertinggi: {target_game_name}")
                print(f"Persyaratan Min: CPU={target_game_min_req.get('CPU_Intel')}/{target_game_min_req.get('CPU_AMD')}, GPU={target_game_min_req.get('GPU_NVIDIA')}/{target_game_min_req.get('GPU_AMD')}/{target_game_min_req.get('GPU_Intel')}, RAM: {target_game_min_req.get('RAM')} GB, Storage: {target_game_min_req.get('File Size')} GB")
                print(f"Persyaratan Rec: CPU={target_game_rec_req.get('CPU_Intel')}/{target_game_rec_req.get('CPU_AMD')}, GPU={target_game_rec_req.get('GPU_NVIDIA')}/{target_game_rec_req.get('GPU_AMD')}/{target_game_rec_req.get('GPU_Intel')}, RAM: {target_game_rec_req.get('RAM')} GB, Storage: {target_game_rec_req.get('File Size')} GB")

            else:
                 print("Tidak ada game relevan yang ditemukan dari query setelah semua upaya.")
                 return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": ["Tidak dapat menemukan game relevan dari query Anda."]})

            laptops_to_evaluate = laptop_df.copy()
            if extracted_budget is not None and extracted_budget > 0:
                print(f"\nMemfilter laptop dengan budget <= {extracted_budget:,}")
                laptops_to_evaluate = laptops_to_evaluate[laptops_to_evaluate['Final Price'] <= extracted_budget].copy()
                print(f"Jumlah laptop setelah filter budget: {len(laptops_to_evaluate)}")
                if laptops_to_evaluate.empty:
                    print("Tidak ada laptop dalam budget yang terdeteksi.")
                    return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": [f"Tidak ada laptop dalam budget {extracted_budget:,} yang tersedia."]})

            if found_laptops_entities:
                print(f"Memfilter laptop berdasarkan entitas: {found_laptops_entities}")
                filter_mask = pd.Series([False] * len(laptops_to_evaluate), index=laptops_to_evaluate.index)
                for entity in found_laptops_entities:
                    if entity in laptop_brand_list:
                        filter_mask |= (laptops_to_evaluate['Brand'].str.lower() == entity.lower())
                    elif entity in laptop_list:
                         filter_mask |= (laptops_to_evaluate['Model'].str.lower() == entity.lower())
                    else:
                        brand_match = process.extractOne(entity, laptops_to_evaluate['Brand'].unique().tolist(), score_cutoff=90)
                        model_match = process.extractOne(entity, laptops_to_evaluate['Model'].tolist(), score_cutoff=90)
                        if brand_match and brand_match[1] >= 90:
                             filter_mask |= (laptops_to_evaluate['Brand'].str.lower() == brand_match[0].lower())
                        if model_match and model_match[1] >= 90:
                             filter_mask |= (laptops_to_evaluate['Model'].str.lower() == model_match[0].lower())

                laptops_to_evaluate = laptops_to_evaluate[filter_mask].copy()
                print(f"Jumlah laptop setelah filter entitas: {len(laptops_to_evaluate)}")
                if laptops_to_evaluate.empty:
                    print(f"Tidak ada laptop yang cocok dengan entitas '{', '.join(found_laptops_entities)}' dalam budget yang terdeteksi.")
                    return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": [f"Tidak ada laptop yang cocok dengan '{', '.join(found_laptops_entities)}' dalam kriteria Anda."]})

            print("\nMengkategorikan laptop berdasarkan persyaratan game target...")
            final_recommendations = categorize_laptops_adapted(
                laptops_to_evaluate,
                target_game_min_req,
                target_game_rec_req
            )

            if final_recommendations.empty:
                print("\nTidak ada rekomendasi laptop yang ditemukan berdasarkan kriteria Anda.")
                print("Tidak ada laptop yang memenuhi persyaratan minimum game yang relevan di antara laptop yang sudah difilter.")
                return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": ["Tidak ada rekomendasi laptop yang memenuhi persyaratan minimum game yang relevan dalam kriteria Anda."]})
            else:
                print("\nHasil Rekomendasi Laptop:")
                if detected_intent == "FIND_BEST_LAPTOP_FOR_GAME":
                    final_recommendations['Category_Order'] = final_recommendations['Category'].apply(
                        lambda x: 0 if x == 'Recommended' else (1 if x == 'Mixed' else 2)
                    )
                    final_recommendations = final_recommendations.sort_values(by=['Category_Order', 'Match_Score'], ascending=[True, False]).drop(columns='Category_Order')
                    print("\nDiurutkan berdasarkan Match Score (Terbaik ke Terburuk) dalam setiap kategori.")
                elif detected_intent == "FIND_CHEAPEST_LAPTOP_FOR_GAME":
                     final_recommendations['Category_Order'] = final_recommendations['Category'].apply(
                        lambda x: 0 if x == 'Recommended' else (1 if x == 'Mixed' else 2)
                     )
                     final_recommendations = final_recommendations.sort_values(by=['Category_Order', 'Final Price', 'Match_Score'], ascending=[True, True, False]).drop(columns='Category_Order')
                     print("\nDiurutkan berdasarkan Harga (Termurah ke Termahal) dalam setiap kategori.")
                else:
                     final_recommendations['Category_Order'] = final_recommendations['Category'].apply(
                        lambda x: 0 if x == 'Recommended' else (1 if x == 'Mixed' else 2)
                     )
                     final_recommendations = final_recommendations.sort_values(by=['Category_Order', 'Match_Score'], ascending=[True, False]).drop(columns='Category_Order')
                     print("\nDiurutkan berdasarkan Match Score (Default).")

                return final_recommendations # Return the DataFrame

        else:
            print("\nIntent tidak dikenali atau tidak didukung saat ini.")
            return pd.DataFrame({"Status": ["Intent Tidak Dikenali"], "Pesan": ["Mohon maaf, niat Anda belum dapat saya proses saat ini."]})


    # --- Buat Embeddings Deskripsi Game ---
    print("Membuat embeddings deskripsi game...")
    descriptions = min_req_df['Description'].fillna('').tolist()
    description_embeddings = model.encode(descriptions, convert_to_tensor=True)
    print("Embeddings berhasil dibuat.")


except FileNotFoundError as e:
    print(f"Error: File data tidak ditemukan. Pastikan file {e.filename} ada di direktori yang sama.")
except Exception as e:
    print(f"Terjadi error saat memuat data atau menyiapkan model: {e}")


@app.route("/recommend", methods=["GET", "POST"])
def recommend_laptop():
    user_query = ""
    if request.method == "POST":
        user_query = request.form.get("query") or (request.json.get("query") if request.is_json else None)
    elif request.method == "GET":
        user_query = request.args.get("query")

    if not user_query:
        return jsonify({"error": "Mohon berikan query untuk rekomendasi laptop."}), 400

    print(f"Menerima query dari user: {user_query}")

    if laptop_df is None or min_req_df is None or rec_req_df is None or model is None or description_embeddings is None:
         return jsonify({"error": "Sistem rekomendasi belum siap. Data atau model gagal dimuat."}), 500


    recommendations_df = get_laptop_recommendations_with_intent(
        user_query, laptop_df, min_req_df, rec_req_df,
        min_req_kb, rec_req_kb,
        model, description_embeddings,
        laptop_list,
        laptop_brand_list,
        unique_keyword_game_map, game_series_map
    )

    # Convert DataFrame to dictionary for JSON response
    recommendations_json = recommendations_df.to_dict('records')

    # Check if the result DataFrame contains an error message
    if not recommendations_df.empty and "Status" in recommendations_df.columns and recommendations_df.iloc[0]["Status"] in ["Informasi Kurang", "Tidak Ditemukan", "Intent Tidak Dikenali"]:
        return jsonify({"query": user_query, "status": recommendations_df.iloc[0]["Status"], "message": recommendations_df.iloc[0]["Pesan"]}), 400 if recommendations_df.iloc[0]["Status"] != "Intent Tidak Dikenali" else 501 # Adjust status code

    return jsonify({"query": user_query, "recommendations": recommendations_json})

if __name__ == '__main__':
    # Ensure the data files and model directory exist before running the app
    if not os.path.exists(LAPTOP_DATA_PATH):
        print(f"Error: File data laptop '{LAPTOP_DATA_PATH}' tidak ditemukan.")
        exit()
    if not os.path.exists(MIN_REQ_DATA_PATH):
        print(f"Error: File data minimum requirements '{MIN_REQ_DATA_PATH}' tidak ditemukan.")
        exit()
    if not os.path.exists(REC_REQ_DATA_PATH):
        print(f"Error: File data recommended requirements '{REC_REQ_DATA_PATH}' tidak ditemukan.")
        exit()
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Direktori model '{MODEL_PATH}' tidak ditemukan.")
        exit()

    # In a production environment, set debug=False
    app.run(debug=True)