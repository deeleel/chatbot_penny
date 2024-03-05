from gpt.utils import *

def get_chatbot_response(query, all_context, count):
    if (count == 0) or (count % 3 == 0):
        all_context = '[C] ' + query + ' [A]'
    else:
        all_context = all_context + ' [C] ' + query + ' [A]'

    all_context = clean_symbols(all_context).replace('[c]', '[C]').replace('[a]', '[A]')
    all_context = ' '.join(all_context.split())
    print(all_context)

    generated_answer, generated = generate_text(query=all_context)
    return generated_answer, all_context + generated_answer