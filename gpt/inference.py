from gpt.utils import *

async def get_chatbot_response(query, all_context, count):
    if (count == 0) or (count % 3 == 0):
        all_context = '[Q] ' + query
    else:
        all_context = all_context + ' [SEP] ' + query


    all_context = clean_symbols(all_context).replace('[q]', '[Q]').replace('[sep]', '[SEP]')
    all_context = ' '.join(all_context.split())
    print(all_context)

    generated_answer, _ = generate_text(query=all_context + '[A]')
    return generated_answer, all_context + ' [SEP] ' + generated_answer



# async def get_chatbot_response_telegram(query, all_messages):
        
    
#         all_context = '[Q] ' + query
#     else:
#         all_context = all_context + ' [SEP] ' + query


#     all_context = clean_symbols(all_context).replace('[q]', '[Q]').replace('[sep]', '[SEP]')
#     all_context = ' '.join(all_context.split())
#     print(all_context)

#     generated_answer, _ = generate_text(query=all_context + '[A]')
#     return generated_answer, all_context + ' [SEP] ' + generated_answer