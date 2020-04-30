import pandas as pd
import numpy as np

class binary_choice:
    
    def __init__(self, data = None, x_cols = None, y_col = None, lr = 0.000001, tol = 1e-7):
        
        self.lr = lr
        self.tol = tol
        self.prob = None
        self.data = data
        self.n = len(data)
        self.x = np.c_[np.ones(self.n), self.data.loc[:,x_cols]]
        self.y = np.expand_dims(np.array(self.data[y_col]), axis = 1)
        self.i, self.log_like, self.w = self.get_estimator(self.x, self.y)
        self.std = self.get_coef_std(self.x, self.w)
        self.MFR_sq, self.NGR_sq = self.get_R_sq()
        
        # Visualizing results
        
        index = ['const'] + x_cols
        self.result = pd.DataFrame(0, index=index, columns=['Coefficient', 'Std. Error', 'Z-value'])
        for i, feat in enumerate(index):
            z_b = self.w[i,0] / self.std[i]
            self.result.loc[feat,'Coefficient'] = round(self.w[i,0], 3)
            self.result.loc[feat,'Std. Error'] = round(self.std[i], 3)
            self.result.loc[feat,'Z-value'] = round(z_b, 3)
    
    def sigmoid(self, x, w):
        
        pred = np.dot(x, w)
        
        return np.where(pred >= 0, 1 / (1 + np.exp(-pred)),
                        np.exp(pred) / (1 + np.exp(pred)))
        
    def update_estimator(self, prob, x, y, w):
        
        grad = np.dot(x.T, y - prob)
        w = w + self.lr * grad
        
        return w
    
    def get_log_likelihood(self, prob, y):
        
        y_1_prob = np.dot(y.T, np.log(prob + 5e-324))
        y_0_prob = np.dot((1 - y).T, np.log(1 - prob + 5e-324))
        log_prob = y_1_prob + y_0_prob
        
        return log_prob
    
    def get_log_likelihood_intercept(self, x, y):
        
        x = x[:,:1]
        y = y
        i, log_like, w = self.get_estimator(x, y)
        
        return log_like
        
    def get_estimator(self, x, y):
        
        i = 1
        w = np.zeros((len(x[0]), 1)) 
        prob = self.sigmoid(x, w)
        log_likelihood = self.get_log_likelihood(prob, y)
        
        while True:
            
            w_iter = self.update_estimator(prob, x, y, w)
            prob_iter = self.sigmoid(x, w_iter)
            log_likelihood_iter = self.get_log_likelihood(prob_iter, y)
            
            #if (log_likelihood >= log_likelihood_iter) or (abs(log_likelihood - log_likelihood_iter) < self.tol):
            if log_likelihood >= log_likelihood_iter:
                break
                
            log_likelihood = log_likelihood_iter.copy()
            w = w_iter.copy()
            prob = prob_iter.copy()
            i += 1
           
        return i, log_likelihood, w
    
    def get_coef_std(self, x, w):
                
        W = np.zeros((x.shape[0], x.shape[0]))

        for i in range(x.shape[0]):
            
            pred = np.dot(w.T, np.expand_dims(x[i], axis = 1))          
            W[i,i] = np.where(pred >= 0, (1 / np.exp(-pred)) / (1 + 1 / np.exp(-pred)) ** 2,
                              np.exp(pred) / (1 + np.exp(pred)) ** 2)

        XtWX_inv = np.linalg.solve(np.dot(np.dot(x.T, W), x), np.identity(x.shape[1]))
        
        return [np.sqrt(XtWX_inv[i,i]) for i in range(len(w))]
    
    def get_R_sq(self):
        
        log_like = self.log_like
        n = self.n
        log_like_intercept = self.get_log_likelihood_intercept(self.x, self.y)
        MFR_sq = 1 - log_like / log_like_intercept
        NGR_sq = (1 - (np.exp(log_like_intercept) / np.exp(log_like)) ** (2 / n)) /\
                 (1 - (np.exp(log_like_intercept)) ** (2 / n))
        
        return MFR_sq, NGR_sq
    
    def __repr__(self):
        return "Regression results:\n\n{}\n\nLog-likelihood: {}\n\nMcFadden R squared: {}\n\nNagelkerke R squared: {}".\
                format(self.result,  round(self.log_like[0,0],3), round(self.MFR_sq[0,0], 3), round(self.NGR_sq[0,0], 3))
