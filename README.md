# Aula-OpenAI
Descrição e tradução da aula DLAi
Sobre este minicurso
Em  ChatGPT Prompt Engineering for Developers,  você aprenderá como usar um modelo de linguagem grande (LLM) para construir rapidamente aplicativos novos e poderosos. Usando a API OpenAI, você poderá criar rapidamente recursos que aprendem a inovar e criar valor de maneiras que antes eram proibitivas em termos de custo, altamente técnicas ou simplesmente impossíveis. Este minicurso ministrado por Isa Fulford (OpenAI) e Andrew Ng (DeepLearning.AI) descreverá como funcionam os LLMs, fornecerá as melhores práticas para engenharia imediata e mostrará como as APIs LLM podem ser usadas em aplicativos para uma variedade de tarefas, incluindo:

Resumindo (por exemplo, resumindo as avaliações dos usuários para fins de brevidade)

Inferir (por exemplo, classificação de sentimento, extração de tópicos)

Transformação de texto (por exemplo, tradução, correção ortográfica e gramatical)

Expandir (por exemplo, escrever e-mails automaticamente)

Além disso, você aprenderá dois princípios-chave para escrever prompts eficazes, como projetar sistematicamente bons prompts e também aprenderá a construir um chatbot personalizado. Todos os conceitos são ilustrados com vários exemplos, que você pode usar diretamente em nosso ambiente de notebook Jupyter para obter experiência prática com engenharia imediata.




# Diretrizes para solicitação
Nesta lição, você praticará dois princípios de prompts e suas táticas relacionadas para escrever prompts eficazes para grandes modelos de linguagem.

Configurar
Carregue a chave API e as bibliotecas Python relevantes.
Neste curso, fornecemos alguns códigos que carregam a chave da API OpenAI para você.

In []:
OpenAI API key for you.

import openai
import openai
import os
​
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
​
openai.api_key  = os.getenv('OPENAI_API_KEY')  


# função auxiliar
Ao longo deste curso, usaremos o modelo gpt-3.5-turbo da OpenAI e o endpoint de conclusão de chat.

Esta função auxiliar tornará mais fácil usar prompts e observar as saídas geradas.
Nota: Em junho de 2023, a OpenAI atualizou o gpt-3.5-turbo. Os resultados que você vê no notebook podem ser ligeiramente diferentes daqueles do vídeo. Alguns dos prompts também foram ligeiramente modificados para produzir os resultados desejados


In []:
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

Nota: Este e todos os outros cadernos de laboratório deste curso usam a biblioteca OpenAI versão 0.27.0.

Para usar a biblioteca OpenAI versão 1.0.0, aqui está o código que você usaria para a função get_completion:


```python
client = openai.OpenAI()

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

## Prompting Principles
- **Principle 1: Write clear and specific instructions**
- **Principle 2: Give the model time to “think”**

### Tactics

#### Tactic 1: Use delimiters to clearly indicate distinct parts of the input
- Delimiters can be anything like: ```, """, < >, `<tag> </tag>`, `:`

In []:
text = f"""
You should express what you want a model to do by \ 
providing instructions that are as clear and \ 
specific as you can possibly make them. \ 
This will guide the model towards the desired output, \ 
and reduce the chances of receiving irrelevant \ 
or incorrect responses. Don't confuse writing a \ 
clear prompt with writing a short prompt. \ 
In many cases, longer prompts provide more clarity \ 
and context for the model, which can lead to \ 
more detailed and relevant outputs.
"""
prompt = f"""
Summarize the text delimited by triple backticks \ 
into a single sentence.
```{text}```
"""
response = get_completion(prompt)
print(response)


#### Tactic 2: Ask for a structured output
- JSON, HTML

In []:

prompt = f"""
Generate a list of three made-up book titles along \ 
with their authors and genres. 
Provide them in JSON format with the following keys: 
book_id, title, author, genre.
"""
response = get_completion(prompt)
print(response)


# Tática 3: Peça ao modelo para verificar se as condições foram satisfeitas

In []:

text_1 = f"""
Making a cup of tea is easy! First, you need to get some \ 
water boiling. While that's happening, \ 
grab a cup and put a tea bag in it. Once the water is \ 
hot enough, just pour it over the tea bag. \ 
Let it sit for a bit so the tea can steep. After a \ 
few minutes, take out the tea bag. If you \ 
like, you can add some sugar or milk to taste. \ 
And that's it! You've got yourself a delicious \ 
cup of tea to enjoy.
"""
prompt = f"""
You will be provided with text delimited by triple quotes. 
If it contains a sequence of instructions, \ 
re-write those instructions in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a sequence of instructions, \ 
then simply write \"No steps provided.\"

\"\"\"{text_1}\"\"\"
"""
response = get_completion(prompt)
print("Completion for Text 1:")
print(response)

In []:

text_2 = f"""
The sun is shining brightly today, and the birds are \
singing. It's a beautiful day to go for a \ 
walk in the park. The flowers are blooming, and the \ 
trees are swaying gently in the breeze. People \ 
are out and about, enjoying the lovely weather. \ 
Some are having picnics, while others are playing \ 
games or simply relaxing on the grass. It's a \ 
perfect day to spend time outdoors and appreciate the \ 
beauty of nature.
"""
prompt = f"""
You will be provided with text delimited by triple quotes. 
If it contains a sequence of instructions, \ 
re-write those instructions in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a sequence of instructions, \ 
then simply write \"No steps provided.\"

\"\"\"{text_2}\"\"\"
"""
response = get_completion(prompt)
print("Completion for Text 2:")
print(response)

# Tática 4: Solicitação de "poucos tiros"

In []:
prompt = f"""
Your task is to answer in a consistent style.

<child>: Teach me about patience.

<grandparent>: The river that carves the deepest \ 
valley flows from a modest spring; the \ 
grandest symphony originates from a single note; \ 
the most intricate tapestry begins with a solitary thread.

<child>: Teach me about resilience.
"""
response = get_completion(prompt)
print(response)

# Princípio 2: Dê tempo ao modelo para “pensar”
Tática 1: Especifique as etapas necessárias para concluir uma tarefa

In []:
text = f"""
In a charming village, siblings Jack and Jill set out on \ 
a quest to fetch water from a hilltop \ 
well. As they climbed, singing joyfully, misfortune \ 
struck—Jack tripped on a stone and tumbled \ 
down the hill, with Jill following suit. \ 
Though slightly battered, the pair returned home to \ 
comforting embraces. Despite the mishap, \ 
their adventurous spirits remained undimmed, and they \ 
continued exploring with delight.
"""
# example 1
prompt_1 = f"""
Perform the following actions: 
1 - Summarize the following text delimited by triple \
backticks with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the following \
keys: french_summary, num_names.

Separate your answers with line breaks.

Text:
```{text}```
"""
response = get_completion(prompt_1)
print("Completion for prompt 1:")
print(response)

# Solicite a saída em um formato especificado

In []:
prompt_2 = f"""
Your task is to perform the following actions: 
1 - Summarize the following text delimited by 
  <> with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the 
  following keys: french_summary, num_names.

Use the following format:
Text: <text to summarize>
Summary: <summary>
Translation: <summary translation>
Names: <list of names in summary>
Output JSON: <json with summary and num_names>

Text: <{text}>
"""
response = get_completion(prompt_2)
print("\nCompletion for prompt 2:")
print(response)

# Tática 2: Instrua o modelo a elaborar sua própria solução antes de chegar a uma conclusão precipitada

In []:
prompt = f"""
Determine if the student's solution is correct or not.

Question:
I'm building a solar power installation and I need \
 help working out the financials. 
- Land costs $100 / square foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost \ 
me a flat $100k per year, and an additional $10 / square \
foot
What is the total cost for the first year of operations 
as a function of the number of square feet.

Student's Solution:
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
"""
response = get_completion(prompt)
print(response)

 # Observe que a solução do aluno na verdade não está correta.
Podemos corrigir isso instruindo o modelo a elaborar primeiro sua própria solução.

In []:
prompt = f"""
Your task is to determine if the student's solution \
is correct or not.
To solve the problem do the following:
- First, work out your own solution to the problem including the final total. 
- Then compare your solution to the student's solution \ 
and evaluate if the student's solution is correct or not. 
Don't decide if the student's solution is correct until 
you have done the problem yourself.

Use the following format:
Question:
```
question here
```
Student's solution:
```
student's solution here
```
Actual solution:
```
steps to work out the solution and your solution here
```
Is the student's solution the same as actual solution \
just calculated:
```
yes or no
```
Student grade:
```
correct or incorrect
```

Question:
```
I'm building a solar power installation and I need help \
working out the financials. 
- Land costs $100 / square foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost \
me a flat $100k per year, and an additional $10 / square \
foot
What is the total cost for the first year of operations \
as a function of the number of square feet.
``` 
Student's solution:
```
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
```
Actual solution:
"""
response = get_completion(prompt)
print(response)

# Limitações do modelo: alucinações
Boie é uma empresa real, o nome do produto não é real.

In []:
prompt = f"""
Tell me about AeroGlide UltraSlim Smart Toothbrush by Boie
"""
response = get_completion(prompt)
print(response)


Notas sobre o uso da API OpenAI fora desta sala de aula
Para instalar a biblioteca OpenAI Python:

!pip instalar openai
A biblioteca precisa ser configurada com a chave secreta da sua conta, que está disponível no site.

Você pode defini-la como a variável de ambiente OPENAI_API_KEY antes de usar a biblioteca:

  !exportar OPENAI_API_KEY='sk-...'
Ou defina openai.api_key com seu valor:

importar openai
openai.api_key = "sk-..."
Uma observação sobre a barra invertida
No curso, estamos usando uma barra invertida \ para fazer o texto caber na tela sem inserir caracteres de nova linha '\n'.
O GPT-3 não é realmente afetado, independentemente de você inserir caracteres de nova linha ou não. Mas ao trabalhar com LLMs em geral, você pode considerar se os caracteres de nova linha no seu prompt podem afetar o desempenho do modelo.

