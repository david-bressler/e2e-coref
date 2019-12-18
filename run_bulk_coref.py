
from bulk_process_coref import Bulk_Coref
import json
import time
import pickle


#names=[ 'Cyber_Crime', \
    # 'Embezzlement'     ,\
    # 'Explosives'       ,\
    # 'Extort_Racketeer_Threats'  ,\
    # 'Fugitive'         ,\
    # 'Human_Trafficking'  ,\
    # 'Kidnapping'       ,\
    # 'Pollution'        ,\
    # 'Smuggling'        ,\
    # 'Stolen_Property' ]

run_typea=2

if run_typea==0: #single at a time

    namea='Stolen_Property' #'drug_traffick'
    print(namea)
    open_filename = '/home/projects/data/' + namea + '.json'
    save_filename = '/home/projects/data/' + namea + '_coref.json'

    with open(open_filename, 'r') as f:
        the_dic = json.loads(f.read())

    n_articles=len(the_dic['results'])
    n_articles=min(1000,len(the_dic['results']))
    #n_articles = 1350
    print('n_articles: ' + str(n_articles) )
    overall_start_time = time.time()

    coreffer = Bulk_Coref()
    the_dic = coreffer.main(the_dic, n_articles)

    with open(save_filename, 'w') as outfile:
        json.dump(the_dic, outfile)

    overall_end_time = time.time() - overall_start_time
    print("OVERALL END TIME: " + str(overall_end_time))
    print("NUMBER OF ARTICLES IS " + str(n_articles))


elif run_typea==1: #multiple

    save_filename = '/home/projects/data/bulk_payload_coref.json'

    the_dic = []
    for line in open('/home/projects/data/cam_sample_data_payload.json', 'r'):
        # the_dic = json.loads(f.read())
        the_dic.append(json.loads(line))

    overall_start_time = time.time()

    coreffer = Bulk_Coref()
    the_dic = coreffer.mult_main(the_dic)

    with open(save_filename, 'w') as outfile:
        json.dump(the_dic, outfile)

    overall_end_time = time.time() - overall_start_time
    print("OVERALL END TIME: " + str(overall_end_time))

elif run_typea==2: #microservices simulation
    # Set up microservice:
    coreffer = Bulk_Coref()
    #Set up data
    namea='Vimpelcom' #'drug_traffick'
    open_filename = '/home/projects/data/' + namea + '.json'
    with open(open_filename, 'r') as f:
        the_dic = json.loads(f.read())
    # Get single document data:
    input_doc=the_dic['results'][0]
    # Process single doc:
    output_doc=coreffer.per_article_proc(input_doc)

