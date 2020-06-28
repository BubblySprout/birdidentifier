
# coding: utf-8

# In[1]:


from fastai import *
from fastai.vision import *


# In[2]:


#urls=Array.from(document.querySelectorAll('.rg_i')).map(el=> el.hasAttribute('data-src')?el.getAttribute('data-src'):el.getAttribute('data-iurl'));
#window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));
#scroll down four times


# In[3]:


URLs.BIRDS = "https://www.kaggle.com/gpiosenka/100-bird-species/download" 


# In[5]:


#tfms = get_transforms(do_flip=False)
#data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=26)


# In[6]:


folder = 'ringnecked'
file = 'urls_ringnecked.txt'


# In[7]:


folder = 'cockatiel'
file = 'urls_cockatiel.txt'


# In[8]:


folder = 'australian'
file = 'urls_australian.txt'


# In[9]:


folder = 'cockatoo'
file = 'urls_cockatoo.txt'


# In[10]:


folder = 'africangrey'
file = 'urls_africangrey.txt'


# In[11]:


folder = 'amazon'
file = 'urls_amazon.txt'


# In[12]:


folder = 'canary'
file = 'urls_canary.txt'


# In[40]:


folder = 'conure'
file = 'urls_conure.txt'


# In[45]:


folder = 'parrotlet'
file = 'urls_parrotlet.txt'


# In[50]:


folder = 'finch'
file = 'urls_finch.txt'


# In[8]:


path = Path('data/birds')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)


# In[52]:


classes = ['cockatiel', 'ringnecked', 'australian', 'cockatoo', 'africangrey', 'amazon', 'canary', 'conure', 'parrotlet', 'finch']


# In[53]:


download_images(path/file, dest, max_pics=200)


# In[20]:


for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_workers=8)


# In[21]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
                                 ds_tfms=get_transforms(), size=224, num_workers=0).normalize(imagenet_stats)


# In[22]:


data.classes


# In[23]:


data.show_batch(rows=6, figsize=(7,8))


# In[ ]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# In[59]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[87]:


learn.fit_one_cycle(4)


# In[76]:


learn.save('stage-1')


# In[77]:


learn.unfreeze()


# In[63]:


learn.lr_find()


# In[80]:


learn.recorder.plot()


# In[91]:


learn.fit_one_cycle(2, max_lr=slice(1e-5,1e-4))


# In[66]:


learn.save('stage-2')


# In[67]:


learn.load('stage-2')


# In[68]:


interp = ClassificationInterpretation.from_learner(learn)


# In[69]:


interp.plot_confusion_matrix()


# In[70]:


from fastai.widgets import *

losses,idxs = interp.top_losses()
top_loss_paths = data.valid_ds.x[idxs]


# In[71]:


db = (ImageList.from_folder(path)
                    .split_none()
                    .label_from_folder()
                    .transform(get_transforms(), size = 224)
                    .databunch()
     )


# In[72]:


learn_cln = cnn_learner(db, models.resnet34, metrics=error_rate)

learn_cln.load('stage-2');


# In[73]:


ds, idxs = DatasetFormatter().from_toplosses(learn_cln)


# In[86]:


ImageCleaner(ds, idxs, path, duplicates=True)


# In[88]:


ImageCleaner(ds, idxs, path, batch_size=10, duplicates=True)

