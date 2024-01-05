import json
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

with open('config/config.json','r') as f:
    config = json.load(f)

def plot_X_y(X,y,cols):
    for col in cols[:-1]:
        plt.scatter(X[col],y)
        plt.ylabel(cols[-1])
        plt.xlabel(col)
        plt.show()

def runModel(model,X_train,X_test,y_train,y_test):
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        r2 = r2_score(y_test,y_pred)
        return r2,y_pred

class EvaluateRegression:
    def __init__(self, X_train,X_test,y_train,y_test):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
    
    def simpleLinear(self,X_train=None,X_test=None):
        if X_train is None:
            X_train=self.X_train
        if X_test is None:
            X_test=self.X_test

        lm = LinearRegression()
        return runModel(lm,X_train,X_test,self.y_train,self.y_test)
    
    def polymomial(self,model=None):

        best = dict(zip(['degree','score','y_pred'],[0,0,None]))
        degrees = range(2,config['max_plynomial_degree']+1) 
        for d in degrees:
            poly = PolynomialFeatures(degree=d)
            Xp_train = poly.fit_transform(self.X_train)
            Xp_test = poly.transform(self.X_test)
            if model is None:
                r2, y_pred = self.simpleLinear(Xp_train,Xp_test)
            else:
                r2, y_pred = runModel(model,Xp_train,Xp_test,self.y_train,self.y_test)

            if r2 > best['score']:
                best['degree'] = d
                best['score'] = r2
                best['y_pred'] = y_pred
        return best
    
    def evaluateRidge(self):
        best = dict(zip(['alpha','degree','score','y_pred'],[0,0,0,None]))
        alphas = config['alpha_range']
        for a in alphas:
            model = Ridge(alpha=a)
            result = self.polymomial(model)
            if result['score'] > best['score']:
                best['alpha'] = a
                best['degree'] = result['degree']
                best['score'] = result['score']
                best['y_pred'] = result['y_pred']

        return best
    
    def evaluateLasso(self):
        best = dict(zip(['alpha','degree','score','y_pred'],[0,0,0,None]))
        alphas = config['alpha_range']
        for a in alphas:
            model = Lasso(alpha=a)
            result = self.polymomial(model)
            if result['score'] > best['score']:
                best['alpha'] = a
                best['degree'] = result['degree']
                best['score'] = result['score']
                best['y_pred'] = result['y_pred']

        return best
    
    def selectBestModel(self):
        best = dict(zip(['model_name','alpha','degree','score','y_pred'],['Simple Linear Regression',0,0,0,None]))
        
        # evaluate Simple Linear Regression Model
        r2,y_pred = self.simpleLinear()
        best['score'] = r2
        best['y_pred'] = y_pred

        # evaluate  Polynomial Regession without Regularization
        results = self.polymomial()
        if results['score'] > best['score']:
            best['model_name'] = 'Polynomial Regession without Regularization'
            best['score'] = results['score']
            best['degree'] = results['degree']
            best['y_pred'] = results['y_pred']

        # evaluate Ridge
        results = self.evaluateRidge()
        if results['score'] > best['score']:
            best['model_name'] = 'Ridge'
            best['score'] = results['score']
            best['degree'] = results['degree']
            best['y_pred'] = results['y_pred']


        # evaluate Lasso
        results = self.evaluateLasso()
        if results['score'] > best['score']:
            best['model_name'] = 'Lasso'
            best['score'] = results['score']
            best['degree'] = results['degree']
            best['y_pred'] = results['y_pred']

        return best