import os
import json


def load_gt(path):
    infor_lst = {}
    with open(path, 'r', encoding='utf8') as f:
        data = json.load(f)['data']
    for question in data:
        question_infor = {
            'questionId': question['questionId'],
            'question': question['question'],
            'answers': question['answers']
        }
        img_path = os.path.basename(question['image'])

        if img_path not in infor_lst.keys():
            new_item = {
                'image': img_path,
                'docId': question['docId'],
                'ucsf_document_id': question['ucsf_document_id'],
                'ucsf_document_page_no': question['ucsf_document_page_no'],
                'questions': [question_infor]
            }
            infor_lst[img_path] = new_item
        else:
            infor_lst[img_path]['questions'].append(question_infor)
    return infor_lst


if __name__ == '__main__':
    gt_path = 'visualize_docvqa/static/val/val_v1.0.json'
    result = load_gt(path=gt_path)
    print(result)


