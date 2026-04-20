import os

replacements = {
    '\"❌ Fortemente sconsigliato (-{penalty}): Contiene generi che eviti ({', '.join(avoided_match)}).\"': '\"❌ Strongly discouraged (-{penalty}): Contains genres you avoid ({', '.join(avoided_match)}).\"',
    '\"✅ Ottimo Match Generi (+{bonus}): Contiene i tuoi generi preferiti ({', '.join(pref_match)}).\"': '\"✅ Great Genre Match (+{bonus}): Contains your favorite genres ({', '.join(pref_match)}).\"',
    '\"🎬 Formato Preferito (+0.8): È un anime {pref_format}.\"': '\"🎬 Preferred Format (+0.8): It is a {pref_format} anime.\"',
    '\"⚠️ Formato Diverso (-0.5): Hai preferito {pref_format}, ma questo è {cand_format}.\"': '\"⚠️ Different Format (-0.5): You preferred {pref_format}, but this is {cand_format}.\"',
    '\"⏱️ Lunghezza Ideale (+0.5): In linea con le tue preferenze sulla lunghezza.\"': '\"⏱️ Ideal Length (+0.5): In line with your length preferences.\"',
    '\"⏳ Lunghezza Sgradita ({l_bonus}): È troppo lungo o troppo corto per i tuoi gusti.\"': '\"⏳ Disliked Length ({l_bonus}): It is too long or too short for your tastes.\"',
    '\"📅 Epoca Preferita (+0.5): Rilasciato nel periodo storico che preferisci ({cand_year}).\"': '\"📅 Preferred Era (+0.5): Released in your preferred historical period ({cand_year}).\"',
    '\"🕰️ Epoca Diversa ({era_bonus}): Rilasciato in un periodo diverso da quello cercato.\"': '\"🕰️ Different Era ({era_bonus}): Released in a different period than what you sought.\"',
    '\"🔍 Tracce Simili (+{implicit_genre_score:.2f}): Ha generi in comune con i tuoi anime preferiti iniziali.\"': '\"🔍 Similar Traces (+{implicit_genre_score:.2f}): Shares genres with your initial favorite anime.\"',
    '\"🎭 Tematiche Affini (+{implicit_tag_score:.2f}): Condivide tematiche forti con i tuoi preferiti (es. {top_matched}).\"': '\"🎭 Similar Themes (+{implicit_tag_score:.2f}): Shares strong themes with your favorites (e.g., {top_matched}).\"',
    '\"🏆 Apprezzato dalla Critica (+{global_bonus:.2f}): L\\'alta qualità globale lo rende una scommessa sicura.\"': '\"🏆 Critically Acclaimed (+{global_bonus:.2f}): High global quality makes it a safe bet.\"',
    '\"📉 Basso Gradimento Globale ({global_bonus:.2f}): La community lo reputa un titolo mediocre.\"': '\"📉 Low Global Rating ({global_bonus:.2f}): The community considers it mediocre.\"',
    '\"📖 Formato Preferito (+0.8): È un {pref_format}.\"': '\"📖 Preferred Format (+0.8): It is a {pref_format}.\"',
    '\"📏 Lunghezza Ideale (+0.5): In linea con le tue preferenze sul numero di capitoli.\"': '\"📏 Ideal Length (+0.5): In line with your chapter count preferences.\"',
    '\"📏 Lunghezza Sgradita ({l_bonus}): Troppi o troppo pochi capitoli per i tuoi gusti.\"': '\"📏 Disliked Length ({l_bonus}): Too many or too few chapters for your taste.\"',
    '\"📅 Epoca Preferita (+0.5): Pubblicato nel periodo storico che preferisci ({cand_year}).\"': '\"📅 Preferred Era (+0.5): Published in your preferred historical period ({cand_year}).\"',
    '\"🕰️ Epoca Diversa ({era_bonus}): Pubblicato in un periodo diverso da quello cercato.\"': '\"🕰️ Different Era ({era_bonus}): Published in a different period than what you sought.\"',
    '\"🔍 Tracce Simili (+{implicit_genre_score:.2f}): Ha generi in comune con i tuoi manga preferiti.\"': '\"🔍 Similar Traces (+{implicit_genre_score:.2f}): Shares genres with your favorite manga.\"',
    '\"CORTO (<14)\"': '\"SHORT (<14)\"',
    '\"MEDIO (14-26)\"': '\"MEDIUM (14-26)\"',
    '\"LUNGO (27+)\"': '\"LONG (27+)\"',
    '\"CLASSICI (Pre-2000)\"': '\"CLASSIC (Pre-2000)\"',
    '\"MODERNI (2000-2015)\"': '\"MODERN (2000-2015)\"',
    '\"RECENTI (Post-2015)\"': '\"RECENT (Post-2015)\"',
    '\"BREVE (<20 cap)\"': '\"SHORT (<20 chap)\"',
    '\"MEDIO (20-100 cap)\"': '\"MEDIUM (20-100 chap)\"',
    '\"LUNGO (100+ cap)\"': '\"LONG (100+ chap)\"',
}

files_to_process = [
    'src/cold_start.py'
]

for file_path in files_to_process:
    if not os.path.exists(file_path):
        continue
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for k, v in replacements.items():
        content = content.replace(k, v)
        
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

print('Pass 3 Translation applied.')
