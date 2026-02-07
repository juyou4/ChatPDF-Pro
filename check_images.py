import json

with open('data/docs/a06bec46742c11712a3c952c6f5a6694.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

content = data['data']['full_text']
print(f'Total chars: {len(content)}')
print(f'Image refs: {content.count("![图片")}')
print(f'\nFirst 1500 chars:')
print(content[:1500])
print(f'\n...\n')
print(f'Last 500 chars:')
print(content[-500:])
