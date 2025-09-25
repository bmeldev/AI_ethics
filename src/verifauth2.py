import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def compare(text1, text2):
    prompt = f'''
Analiza si los siguientes dos textos en español fueron escritos por la misma persona. Haz el análisis basado tanto en el estilo como el contenido del texto, pero sobre todo en el estilo.
Responde solo con un número entre 0 y 1 (donde 0 significa 'diferente autor' y 1 significa 'mismo autor').

Texto 1: '{text1}'
Texto 2: '{text2}'

Escribe solo el número, sin explicación, con un máximo de 2 decimales.
'''
    response = client.responses.create(
        model='gpt-4.1-mini',
        input=prompt,
        temperature=0
    )

    output = response.output_text.strip()
    try:
        probability = float(output)
    except ValueError:
        raise ValueError(f'Error output model: {output}')
    return probability
