from Flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

# initialize new Flask instance with argument __name__ (so Flask knows HTML template folder
# is in same directory as Flask is located)
app = Flask(__name__)

# specify url that triggers execution of the function "home"; the "home" function renders 
# the home.html file (located in "templates" folder)
@app.route('/')
def home():
    return render_template('home.html')

# within the predict function, we do the following: access the True and Fake datasets, pre-process
# the text, make predictions, store the model, access message entered by the user, use our model 
# to make prediction for its label
# the POST method transports form data to the server in the message body

@app.route('/predict', methods=['POST'])
def predict():
    df=pd.read_csv("True.csv", encoding="latin-1")
    #Features and labels
    df['label'] = df['class'].map({'True' : 0, 'False' : 1})
    x = df['message']
    y = df['label']

    #Extract feature with CountVectorizer
    cv = CountVectorizer()
    x = cv.fit_transform(x)     #Fitting the data
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state = 42)

    ## need to check on whether need to add clf. fit etc code
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        prediction = clf.predict(vect)
    return render_template('result.html', prediction = prediction)

# the if __name == '__main__' statement ensure that the run function will only run the application 
# on the server when the script is directly executed by Python interpreter
if __name__ == '__main__':
    app.run(debug=True)

