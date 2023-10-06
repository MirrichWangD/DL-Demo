# coding: utf-8

#导入所需的模块
import urllib.request
import os
import tarfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer


""" 
# 自动导入IMDB数据集
imdb = tf.keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data() 
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
train_text = x_train
y_train = y_train
test_text = x_test
y_test = y_test
""" 


#手动导入数据集
url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath="19DL/IMDB/aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):
    result=urllib.request.urlretrieve(url,filepath)
    print('downloaded:',result)

if not os.path.exists("19DL/IMDB/aclImdb"):
    tfile = tarfile.open("19DL/IMDB/aclImdb_v1.tar.gz", 'r:gz')
    result=tfile.extractall('19DL/IMDB/')

#使用正则表达式删除HTML的标签
import re
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)

#使用read_files()函数读取IMDB文件目录
def read_files(filetype):
    path = "19DL/IMDB/aclImdb/"
    file_list=[]

    positive_path=path + filetype+"/pos/"
    for f in os.listdir(positive_path):
        file_list+=[positive_path+f]
    
    negative_path=path + filetype+"/neg/"
    for f in os.listdir(negative_path):
        file_list+=[negative_path+f]
        
    print('read',filetype, 'files:',len(file_list))
       
    all_labels = ([1] * 12500 + [0] * 12500) 
    
    all_texts  = []
    for fi in file_list:
        with open(fi,encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
            
    return all_labels,all_texts


# 导入的数据
y_train,train_text=read_files("train")
y_test,test_text=read_files("test")

# 查看数据，0-12499项：正面评价文字，0-12499项：正面评价，全是“1”。
# 12500-24999：负面评价文字，12500-24999：负面评价,全是”0”
len(train_text)
len(test_text)
train_text[0]
y_train[0]

train_text[12501]
y_train[12501]


# 建立token，即用训练的25000评价文字产生一个字典，
# 只取排序后的前2000名英文单词进入字典（也可以取更大的数进入字典）
token = Tokenizer(num_words=4000)
token.fit_on_texts(train_text)


# 使用token字典，将“影评文字”转为“数字列表”
# 将每一篇文章的文字转换一连串的数字，只有在字典中的文字才会被转换为数字
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq  = token.texts_to_sequences(test_text)


# 让转换后的数字长度相同100
x_train = sequence.pad_sequences(x_train_seq, maxlen=400)
x_test  = sequence.pad_sequences(x_test_seq,  maxlen=400)


"""-----------------------------+
# 建立长短期记忆神经网络LSTM模型
-------------------------------+
"""
model = tf.keras.Sequential()

#嵌入层 (字典vocab_size=4000，另外将每一个数字映射到64维的向量空间中去)
model.add(tf.keras.layers.Embedding(output_dim=64,
                    input_dim=4000, 
                    input_length=400))
model.add(tf.keras.layers.Dropout(0.2))

""" # 长短期记忆神经网络LSTM """
model.add(tf.keras.layers.LSTM(32))

#隐藏层，有256个神经元
model.add(tf.keras.layers.Dense(units=256,
                activation='relu' ))
model.add(tf.keras.layers.Dropout(0.2))

#输出层，一个神经元，用Sigmoid函数作激活函数，预测 0，1变量的概率。
# 最后输出 0或者1的概率。
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

#LSTM模型框架摘要
model.summary()



# 训练前，要将list：y_train和y_test转化为numpy.ndarray
# 否则会出错  
# 'int'>, <class 'int'>, <class 'int'>, <class 'int'>, <class 'int'>]
type(x_train)
type(y_train)
y_train = np.array(y_train)

type(x_test)
type(y_test)
y_test = np.array(y_test)


# # 训练模型
model.compile(loss='binary_crossentropy', 
              optimizer='rmsprop', 
              metrics=['accuracy'])

train_history =model.fit(x_train, y_train,batch_size=100, 
                         epochs=10,verbose=2,
                         validation_split=0.2)


# 可视化
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

show_train_history(train_history,'accuracy','val_accuracy') 
show_train_history(train_history,'loss','val_loss')


# # 评估模型的准确率  86.67%
scores = model.evaluate(x_test, y_test, verbose=1)
scores[1]



# # 预测概率
probility=model.predict(x_test)
probility[:10]

for p in probility[12500:12510]:
    print(p)


# # 预测结果
predict=model.predict_classes(x_test)
predict[:10]

predict_classes=predict.reshape(-1)
predict_classes[:10]


# # 创建一个函数，查看预测结果
SentimentDict={1:'正面的',0:'负面的'}
def display_test_Sentiment(i):
    print(test_text[i])
    print('标签label:',SentimentDict[y_test[i]],
          '预测结果:',SentimentDict[predict_classes[i]])
display_test_Sentiment(2)

predict_classes[12500:12510]
display_test_Sentiment(12502)
display_test_Sentiment(12504)



#随意给出一段影评，用刚才建立的ANN模型预测新的影评（正面或负面）
# 将影评储存在input_text中
input_text='''
Oh dear, oh dear, oh dear: where should I start folks. I had low expectations already because I hated each and every single trailer so far, but boy did Disney make a blunder here. I'm sure the film will still make a billion dollars - hey: if Transformers 11 can do it, why not Belle? - but this film kills every subtle beautiful little thing that had made the original special, and it does so already in the very early stages. It's like the dinosaur stampede scene in Jackson's King Kong: only with even worse CGI (and, well, kitchen devices instead of dinos).
The worst sin, though, is that everything (and I mean really EVERYTHING) looks fake. What's the point of making a live-action version of a beloved cartoon if you make every prop look like a prop? I know it's a fairy tale for kids, but even Belle's village looks like it had only recently been put there by a subpar production designer trying to copy the images from the cartoon. There is not a hint of authenticity here. Unlike in Jungle Book, where we got great looking CGI, this really is the by-the-numbers version and corporate filmmaking at its worst. Of course it's not really a "bad" film; those 200 million blockbusters rarely are (this isn't 'The Room' after all), but it's so infuriatingly generic and dull - and it didn't have to be. In the hands of a great director the potential for this film would have been huge.
Oh and one more thing: bad CGI wolves (who actually look even worse than the ones in Twilight) is one thing, and the kids probably won't care. But making one of the two lead characters - Beast - look equally bad is simply unforgivably stupid. No wonder Emma Watson seems to phone it in: she apparently had to act against an guy with a green-screen in the place where his face should have been. 
'''

input_seq = token.texts_to_sequences([input_text])
len(input_seq[0])

pad_input_seq  = sequence.pad_sequences(input_seq , maxlen=100)
len(pad_input_seq[0])

predict_result=model.predict_classes(pad_input_seq)
predict_result[0][0]
SentimentDict[predict_result[0][0]]


# 创建一个函数，直接处理新的影评文字，然后输出预测结果
def predict_review(input_text):
    input_seq = token.texts_to_sequences([input_text])
    pad_input_seq  = sequence.pad_sequences(input_seq , maxlen=100)
    predict_result=model.predict_classes(pad_input_seq)
    print(SentimentDict[predict_result[0][0]])


# 查看《美女与野兽》的影评
#http://www.imdb.com/title/tt2771200/
#http://www.imdb.com/title/tt2771200

predict_review('''
It's hard to believe that the same talented director who made the influential cult action classic The Road Warrior had anything to do with this disaster.
Road Warrior was raw, gritty, violent and uncompromising, and this movie is the exact opposite. It's like Road Warrior for kids who need constant action in their movies.
This is the movie. The good guys get into a fight with the bad guys, outrun them, they break down in their vehicle and fix it. Rinse and repeat. The second half of the movie is the first half again just done faster.
The Road Warrior may have been a simple premise but it made you feel something, even with it's opening narration before any action was even shown. And the supporting characters were given just enough time for each of them to be likable or relatable.
In this movie there is absolutely nothing and no one to care about. We're supposed to care about the characters because... well we should. George Miller just wants us to, and in one of the most cringe worthy moments Charlize Theron's character breaks down while dramatic music plays to try desperately to make us care.
Tom Hardy is pathetic as Max. One of the dullest leading men I've seen in a long time. There's not one single moment throughout the entire movie where he comes anywhere near reaching the same level of charisma Mel Gibson did in the role. Gibson made more of an impression just eating a tin of dog food. I'm still confused as to what accent Hardy was even trying to do.
I was amazed that Max has now become a cartoon character as well. Gibson's Max was a semi-realistic tough guy who hurt, bled, and nearly died several times. Now he survives car crashes and tornadoes with ease?
In the previous movies, fuel and guns and bullets were rare. Not anymore. It doesn't even seem Post-Apocalyptic. There's no sense of desperation anymore and everything is too glossy looking. And the main villain's super model looking wives with their perfect skin are about as convincing as apocalyptic survivors as Hardy's Australian accent is. They're so boring and one-dimensional, George Miller could have combined them all into one character and you wouldn't miss anyone.
Some of the green screen is very obvious and fake looking, and the CGI sandstorm is laughably bad. It wouldn't look out of place in a Pixar movie.
There's no tension, no real struggle, or any real dirt and grit that Road Warrior had. Everything George Miller got right with that masterpiece he gets completely wrong here. 
''')


predict_review('''
Sure, I'm a huge film snob who (on the surface) only likes artsy-fartsy foreign films from before the 60's, but that hasn't stopped me from loving Disney's Beauty & The Beast; in fact, it's probably my favorite American animated film and is easily Disney's finest work. It's beautiful, it's breathtaking, it's warm, it's hilarious, it's captivating, and, in Disney fashion, it's magical. When I learned that Disney would be remaking their classic films, B&TB was undeniably the best wrapped package. How could they go wrong?
Oh man, they went wrong.
First thing's first: this film is so flat. The directing was dull and uninteresting throughout the entire film and it honestly felt like one of the Twilight sequels...and then I looked it up and found out that, yes, director Bill Condon was the man behind Breaking Dawn parts 1 & 2. Every shot looks bored and uninterested, which contrasts heavily with the original animated film that was constantly popping with vibrancy. The script too is boring because it's almost a complete remake of the original, though I guess most people won't mind that.
Next: the CGI is horrid. Although I didn't care for The Jungle Book from last year, I could at least admit that the CGI was breathtaking. The same cant be said for this film. Characters like Lumière, Cogsworth, Mrs Potts, and most of the cursed appliances have very strange, lifeless faces that are pretty off putting to be looking at for such a long time. All of the sets too look artificial and fake, especially the town towards the beginning. However, the biggest offender is easily and infuriatingly the character that mattered most: The Beast. The CGI on the Beast's face is so distracting that it completely takes you out of the film. His eyes are completely devoid of soul, and his mouth is a gaping video game black hole of fiction. Klaus Kinski looked much better in the Faerie Tale Theatre episode of Beauty & The Beast, and that was a 1984 TV show episode. But do you know why it looked better? Because it was an actual face with actual eyes, not some video game computerized synthetic monstrosity. When will studios learn that practical effects will always top CGI?
Finally: wasted casting. Emma Watson is beautiful, but she's no Belle. She is completely devoid of the warmth and humanity that made the animated Belle so beloved. Instead, she is cold and heartless throughout most of the film. Kevin Kline is 100% wasted and does nothing except look old. Ian McKellan, Ewan McGregor, Emma Thompson, and even Dan Stevens as the Beast are very expendable and could've been played by anyone else. The only good characters are Gaston and LeFou, mostly because they are fun and played by actors who breathe new life into their original shapes. If anything, this film should've been about Gaston and LeFou, but that would never happen because that would mean Disney couldn't cater to blind nostalgic 90's kids.
Overall, this film is a complete bore. It could've been better if even the special effects were good, but the CGI in particular is horrendous. I'm all for Disney remaking their nostalgia- catering 90's films, but they need to be interesting. This film, sadly, is not. Even the Christmas sequel is better than this film because it's at least something. 
''')


predict_review('''
I was really looking forward to this film. Not only has Disney recently made excellent live-action versions of their animated masterpieces (Jungle Book, Cinderella), but the cast alone (Emma Watson, Ian McKellen, Kevin Kline) already seemed to make this one a sure hit. Well, not so much as it turns out.
Some of the animation is fantastic, but because characters like Cogsworth (the clock), Lumière (the candelabra) and Chip (the little tea cup) now look "realistic", they lose a lot of their animated predecessors' charm and actually even look kind of creepy at times. And ironically - unlike in the animated original - in this new realistic version they only have very limited facial expressions (which is a creative decision I can't for the life of me understand).
Even when it works: there can be too much of a good thing. The film is overstuffed with lush production design and cgi (which is often weirdly artificial looking though) but sadly lacking in charm and genuine emotion. If this were a music album, I'd say it is "over-produced" and in need of more soul and swing. The great voice talent in some cases actually seems wasted, because it drowns in a sea of visual effects that numbs all senses. The most crucial thing that didn't work for me, though, is the Beast. He just never looks convincing. The eyes somehow don't look like real eyes and they're always slightly off.
On the positive side, I really liked Gaston, and the actor who played him, Luke Evans, actually gave the perhaps most energized performance of all. Kevin Kline as Belle's father has little to do but to look fatherly and old, but he makes the most of his part. Speaking of Belle, now that I've seen the film, I think her role was miscast. I think someone like Rachel McAdams would actually have been a more natural, lively and perhaps a bit more feisty Belle than Emma Watson.
If you love the original, you might want to give this one a pass, it's really not that good (although at least the songs were OK). Also, I'd think twice before bringing small children; without cute animated faces, all those "realistic" looking creatures and devices can be rather frightening for a child. ''')


predict_review('''
Up front: I'm probably not the right audience for this film. I only went because I was invited, and I wouldn't have gone to check this one out otherwise.
Firstly, some of the production values are really beautiful and reminded me of the animated classic in a good way. Also, the voice cast for the clock and the kitchen devices are great.
Secondly, the actors, well... this may sound kind of harsh, but I've never seen Emma Watson act so stiff in a movie. Her performance is wooden, which is pretty bad considering she's supposed to be the heart of the film. Also, she probably won't start a singing career anytime soon.
Thirdly (and most importantly), Beast. That's where they really dropped the ball. Giving him a lifeless CGI face was an unforgivable mistake, and it's such a constant distraction that I could never really get into the movie.
Overall, I'm afraid I wouldn't recommend this movie, at least not to adults. I'm sure most kids would enjoy it though, and it's not really a bad film: just a very mediocre one. 6 stars out of 10. 
''')


predict_review('''
Full disclosure, I didn't think the first movie was as bad as it was made out to be. It wasn't good in almost any sense, but it was to be expected given the combination of source material, resources and constraints.
That said, this sequel is 20x better than the first. Having established the characters in the first movie, the actors seem to be able to act now comfortably in their parts. The story becomes much more nuanced with plenty of dynamics on the go.
SPOILERS from now on
Can they maintain a "vanilla" relationship? Is he going to become controlling again and ruin things? Will she let it get out of control and ruin things also or stay on it? Who is that stalky girl and what happened to her exactly? what about his mother? and that ex of his? Will something occur with her infatuated boss?
On top of all of this, I realised while watching that the series was never about a bizarre sadist control freak, it's actually about all men and the story of a woman trying to find the balance between accepting or desiring the dominant behaviour of the male archetype and maintaining strength and independence in such a relationship.
While of course the fact that he is rich, while possibly relating to the power struggle, looks like it is going to be more and more used for generating further drama. The romance is much more evident in this movie to/ 
''')


#A Star Wars Story (2018) 
# Filter by Rating: 10/10
predict_review('''
I grew up with the original Star Wars trilogy when it was new and fresh, and Harrison Ford has been my favorite actor since 1977. At first I was skeptical to having a young Han Solo movie, but now I must say it´s a great adventure movie! 
Taking place about 10-15 years before Star Wars A New Hope, it does of course show us how Han Solo met Chewbacca, and so on - and it´s both interesting and highly entertaining to see exactly how it happened! Alden Ehrenrich is a perfect casting choice as the young Han Solo, and this is coming from a big Harrison Ford fan that at first was hesitating - but gosh, one has to relax a little and just enjoy it all! He is great! As Harrison Ford said himself when watching it: "He nailed it!"
''')

# Filter by Rating: 10/10
predict_review('''
Tonight I went to watch Solo: A Star Wars Story and all I can say is that I was incredibly entertained and that I enjoyed every bit of the film. I am not going to spoil anything, but it's one of those moments where I felt like I was really experiencing a Star Wars story. I did not like 'The Force Awakens' neither did I like 'The Last Jedi' much, but Solo blew them all away! It goes above 'Rogue One' for me, as it cuts down on 'annoying things' like running after a child to save her in middle of the gunfire and the already seen moments. A lot of twists and turns in 'Solo: A Star Wars Story', moments you'd not expect and it's also quite dark, as it is set in the gloomy time where Empire controlled majority of the places.
Parents and I have thoroughly enjoyed it and we finally felt that old Star Wars vibe again. Don't let the hate comments sway you, go see it and conclude yourself. It's packed with constant action, it never felt boring!
''')


# Filter by Rating: 1/10
predict_review('''
The rating for this piece of s*it is negative infinity over 10. 
Star wars is dead. Before "Solo", I thought that "The Last Jedi" was the worst film in the Star Wars franchise. However, "Solo" is ways worse than "The Last Jedi". Actually I should never call "Solo" a film, as there is neither story nor character development. Although the performances, music score and action sequences are acceptable, say 4/10, they are mindless and meaningless, and without a connection to the star wars series. The actor who plays Han Solo, is not bad, despite the fact I don't know his name. For now let's call him Actor A. Actor A's line delivery is decent and worths at least 5/10. However, the charisma of Actor A is zero, and cannot even be compared with Harrison Ford. This is a wrong casting. 
Disney is dead. Previously, Disney has almost never, if not never, made an Oscar-winning film, and has made essentially, if not absolutely, zero contribution to the art of filmmaking. This film has further proved that embarrassing statement. There is simply no point in making "Solo". Disney is greedy as always, and just tries to make the maximum money with the least effort. "Solo" just adds nothing to star wars.
Ron Howard is dead. The guy who made "The beautiful mind" has long gone. Perhaps someone would say that he took over the project halfway, but that is not an excuse for making "Solo" so bad. 
Let's pray that Pixar will not be ruined by Disney.
''')

# Filter by Rating: 1/10
predict_review('''
Are there no more talented writers left in Hollywood? Is the problem that all of you Disney "artists" grew up on Prozac and video games?
What was once a franchise that had deep spiritual qualities is now nothing more than a Saturday morning cartoon.
Movie after movie I kept thinking "this will be the one that gets us back on track" but each post-Lucas release is worse than the one before it!!!!
Tomorrow I will be selling all of my SW collectables online with the exception of any item related to episodes 1-6.
Good luck in the future, Disney. (You're going too need it.)
''')



# Filter by Rating: 6/10
predict_review('''
SPOILER FREE REVIEW
This movie may be struggling in my mind from Star Wars Overexposure and I'm aware of that. However, it doesn't change that it genuinely felt as though everyone (except for Donald Glover) had never watched any of the previous Star Wars films. Han Solo is one of the most badass characters in the history of cinema. He's fun, unapologetic, and badfreakingass! I just thought Ron Howard dropped the ball in creating a fun movie and Alden dropped the ball on his portrayal of Solo. The movie just felt wildly underwhelming, lazy, and rushed. I struggled to stay awake for this.
''')


# Filter by Rating: 4/10
predict_review('''
Certain surprises can be a nice thing to add a little twist to the plot. Ron Howard on the other hand added a surprise that killed the Lucas prequel timeline. 
I will give a hint by saying Solo in this movie is around 20 years old. In the New Hope, Solo is 30's and Ben Kenobi is in his 60's. And lastly, Kenobi was 20 years old in Phantom Menece.
''')

# Filter by Rating: 4/10
predict_review('''
Sorry to say but the best part of the movie was the music during the credits. All this movie consisted of was a bunch of third rate crime bosses and shifting allegiances. People would show up from another part of the movie with no explanation. This was supposed to be about Solo which I assumed would contain more flying. I was wrong. I'd love to see footage by the original director. The trailer is considerably better than the movie; quite exciting and splashy. The movie was neither.
''')



# # serialize model to JSON 和 h5
model_json = model.to_json()
with open("19DL/IMDB/Keras_IMDB_LSTM_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("19DL/IMDB/Keras_IMDB_LSTM_model.h5")
print("Saved model to disk")

