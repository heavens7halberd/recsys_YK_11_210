import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, KNNBasic

def load_data(path, n_users=100, n_events=500, sample_size=500):
    df = (pd.read_csv(path)
          .fillna('')
          .assign(text=lambda x: x[['Product Name','Brand Name','About Product','Product Description']]
                  .agg(' '.join, axis=1))
          .query("text.str.strip()!=''"))
    np.random.seed(42)
    users = [f"U{i}" for i in range(n_users)]
    items = df['Uniq Id'].sample(min(sample_size, len(df))).tolist()
    events = pd.DataFrame({
        'user_id': np.random.choice(users, n_events),
        'item_id': np.random.choice(items, n_events),
        'rating': np.random.choice([0.5,1], n_events, p=[0.7,0.3])
    })
    return df, events

class CBRecommender:
    def __init__(self, items, text='text'):
        self.ids = items['Uniq Id'].values
        M = TfidfVectorizer(stop_words='english', max_features=2000).fit_transform(items[text])
        self.sim = cosine_similarity(M)

    def recommend(self, item_id, k=5):
        i = np.where(self.ids == item_id)[0][0]
        idx = np.argsort(-self.sim[i])
        return self.ids[idx[idx != i][:k]]

class CFRecommender:
    def __init__(self, events):
        reader = Reader(rating_scale=(0, 1))
        data = Dataset.load_from_df(events[['user_id','item_id','rating']], reader).build_full_trainset()
        self.model = KNNBasic(sim_options={'name':'cosine','user_based':True})
        self.model.fit(data)

    def recommend(self, user_id, events, k=5):
        seen = set(events.query('user_id==@user_id').item_id)
        all_i = set(events['item_id'])
        preds = []
        for i in all_i - seen:
            pred = self.model.predict(user_id, i)
            preds.append((i, pred.est))
        preds_sorted = sorted(preds, key=lambda x: -x[1])
        return [i for i, _ in preds_sorted[:k]]

class HeuristicRecommender:
    @staticmethod
    def popular(events, k=5):
        return events.query('rating==1').item_id.value_counts().index[:k].tolist()

def get_product_info(items, item_ids):
    return items[items['Uniq Id'].isin(item_ids)][['Uniq Id', 'Product Name']]

if __name__ == '__main__':
    items, events = load_data('data.csv')

    cb = CBRecommender(items)
    cb_recs = cb.recommend(item_id=items['Uniq Id'].iloc[0])
    print("Content-Based рекомендации:")
    print(get_product_info(items, cb_recs))

    cf = CFRecommender(events)
    cf_recs = cf.recommend(user_id='U1', events=events)
    print("\nCollaborative рекомендации:")
    print(get_product_info(items, cf_recs))

    print("\nПопулярные товары:")
    popular_recs = HeuristicRecommender.popular(events)
    print(get_product_info(items, popular_recs))
