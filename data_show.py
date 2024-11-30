import os
import pandas as pd
import matplotlib.pyplot as plt

image_dir = 'C:\\Users\\drago\Desktop\\의료 인공지능\\train-image\\image'
metadata_path = 'C:\\Users\\drago\\Desktop\\의료 인공지능\\train-metadata.csv'

metadata = pd.read_csv(metadata_path)

# 결측치(빈칸) 제거 및 열 선택
tumor_location = metadata.dropna(subset=['anatom_site_general'])['anatom_site_general']
mal_ben = metadata['target']

# csv 파일에 어떤 부위가 있는지 확인
print(f'location of tumor: {tumor_location.unique()}')

# 부위별로 benign(0)과 malignant(1)가 몇 개 있는지 확인
print(metadata.groupby(['anatom_site_general', 'target']).size())

# 그래프 그리기
grouped_data = metadata.dropna(subset=['anatom_site_general']).groupby(['anatom_site_general', 'target']).size()
ax = grouped_data.plot(kind='bar', figsize=(10, 6), stacked=True)

# 그래프 꾸미기
plt.title('Benign (0) vs Malignant (1) at tumor location')
plt.xlabel('tumor location')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Target', labels=['Benign (0)', 'Malignant (1)'])
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 그래프 표시
plt.tight_layout()
plt.show()
