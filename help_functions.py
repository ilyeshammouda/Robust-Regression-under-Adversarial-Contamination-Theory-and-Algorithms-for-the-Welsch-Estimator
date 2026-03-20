import numpy as np
from scipy.linalg import inv, det
from scipy.stats import invweibull,burr12,t,bernoulli
import statsmodels.api as sm
from statsmodels.robust.norms import TukeyBiweight,Hampel
from sklearn.model_selection import KFold, train_test_split
from statsmodels.robust.robust_linear_model import RLM
from tqdm import tqdm



def generate_linear_model(n, p,beta_etoile,x_mean=0,x_std=1,noise_mean=0,noise_std=1 ,
                         noise_type='gaussian',df=5,seed=10,k=4,outliers_perc=0,outliers=False,outlier_const=1000,var_pareto=1000):
    np.random.seed(seed)

    X = np.random.normal(loc=x_mean, scale=x_std,size=(n, p))
    
    outiliers_vec=np.zeros(n)

    if outliers:
        numb_outliers=int(n*outliers_perc) # Il faut que le pourcentage des outiliers soit entre 0 et 1
        indices_outliers = np.random.choice(n, numb_outliers, replace=False)
        #valeurs_outliers = outlier_const*np.random.choice([-1, 1], numb_outliers)
        valeurs_outliers = outlier_const*np.random.choice([x for x in range(-9, 10) if x != 0], numb_outliers) 
        outiliers_vec[indices_outliers] = valeurs_outliers


    if noise_type == 'gaussian':
        epsilon = np.random.normal(loc=noise_mean,scale=noise_std, size=n)


    elif noise_type == 'student':
        epsilon = t.rvs(df=df, size=n)  # df est le nombre de degrés de liberté pour la t de Student
        epsilon=epsilon/np.sqrt(np.var(epsilon))


    elif noise_type=='pareto':
        pareto=np.random.pareto(a=k,size=n)
        epsilon=pareto - np.mean(pareto)
        epsilon=epsilon/np.sqrt(np.var(epsilon)) # Je centre et normalise le bruit 

    elif noise_type=='log-normlal':

        epsilon=np.random.lognormal(mean=0, sigma=1, size=n)
    
    elif noise_type=='burr':

        epsilon = np.random.gamma(100, 30, size=n)
        epsilon = (epsilon - np.mean(epsilon)) / np.std(epsilon)

    else:
        raise ValueError("Type de bruit non supporté : choisir 'Gaussian' ou 'Student' ou 'Pareto' ")
    
    # Y vérifie le modèle linéaire classique: Y=X*beta+epsilon+outliers
    Y = X @ beta_etoile + epsilon+outiliers_vec
    
    return Y,X,outiliers_vec


def generate_corrupted_model(n, p,beta_etoile,x_mean=0,x_std=1,noise_mean=0,noise_std=1 ,
                         noise_type='gaussian',df=5,seed=10,k=4,outliers_perc=0,outliers=False,outlier_const=1000):
    np.random.seed(seed)

    X = np.random.normal(loc=x_mean, scale=x_std,size=(n, p))
    
    outiliers_vec=np.zeros(n)

    if outliers:
        numb_outliers=int(n*outliers_perc) # Il faut que le pourcentage des outiliers soit entre 0 et 1
        indices_outliers = np.random.choice(n, numb_outliers, replace=False)
        valeurs_outliers = outlier_const*np.random.choice([-1, 1], numb_outliers) 
        outiliers_vec[indices_outliers] = valeurs_outliers


    if noise_type == 'gaussian':
        epsilon = np.random.normal(loc=noise_mean,scale=noise_std, size=n)


    elif noise_type == 'student':
        epsilon = t.rvs(df=df, size=n)  # df est le nombre de degrés de liberté pour la t de Student
        epsilon=epsilon/np.sqrt(np.var(epsilon))


    elif noise_type=='pareto':
        pareto=np.random.pareto(a=k,size=n)
        epsilon=pareto - np.mean(pareto)
        epsilon=epsilon/np.sqrt(np.var(epsilon)) # Je centre et normalise le bruit 
    else:
        raise ValueError("Type de bruit non supporté : choisir 'Gaussian' ou 'Student' ou 'Pareto' ")
    
    # Y vérifie le modèle linéaire classique: Y=X*beta+epsilon+outliers
    Y = X @ beta_etoile + epsilon+outiliers_vec
    
    # Corruption de X. Je choisi le modèle statistique comme dans l'articel:
    # Au lieu d'observer X, on observe Z=X+UW

    bernoulli_diagonale= np.random.choice([-1, 1], n)
    U = np.diag(bernoulli_diagonale)
    W=np.random.normal(loc=0, scale=30,size=(n, p))
    Z=X+np.dot(U,W)
    return Y,Z










def alpha_divergence(tau,x):
    return (1/tau)*(1-np.exp((-tau/2)*(x**2)))


def huber(u,tau):
        
        if abs(u)<=tau:
            return (1/2)*(u**2)
        else:
            return tau*(abs(u)-(1/2)*(tau))


def alpha_div_regression(tau,X,Y,beta):
    n=Y.shape[0]
    return (1/n) * np.sum(alpha_divergence(tau,Y - X @ beta))




def alpha_div_somme_2(tau,x1,x2):
    return (1/2)*(alpha_divergence(tau,x1)+alpha_divergence(tau,x2))


def Huber_somme_2_vriables(tau,x1,x2):
    return (1/2)*(huber(x1,tau)+huber(x2,tau))


def huber_regression(tau,X,Y,beta):
    n=Y.shape[0]
    return (1/n) * np.sum(huber(Y - X @ beta,tau))






class alpha_divergence_tools:
    '''
    Dans cette classe j'implimente quelques outils en relation avec la divergence alpha.
    Typiquement, la loss qu'on considére, ainsi que le gradient de celle-ci par rapport à beta.
    '''
    def __init__(self,X,y,sigma=1):
        self.X= X
        self.y= y
        self.sigma= sigma
        self.n,self.p= X.shape

    def alpha_divergence_weight(self,beta,tau):
        '''
        Avec cette fonction, je calcule les 'poids'. Dans la fonction, cette partie correspond à la partie 
        exponentielle de la divergence alpha.
        '''
        v = ((self.y - np.dot(self.X, beta)) / self.sigma) ** 2
        return np.exp(-tau * v / 2)



    def alpha_divergence_loss(self,beta,tau):
        '''
        Ici je calcule la divergence alpha pour les X et Y donnés. 
        Cette fonction est en effet à seule variable, ça sera utile dans la suite de la considérer ainsi.
        '''
        ones= np.ones(self.n)
        v_exp= self.alpha_divergence_weight(beta,tau)
        v_inter=ones-v_exp
        return(1/(self.n*tau))*v_inter.sum()

    def function_fixed_point(self,beta,tau):
        '''
        Là je définis la fonction objective que je veux résoudre avec la méthode du point fixe.
        Pour les méthode scipy il est important de la considérer comme une fonction à une seule variable.
        '''

        weight= self.alpha_divergence_weight(beta,tau)
        term1= np.dot(((self.X.T) * weight.reshape(1, -1)),self.X)
        itermediate=self.y*weight
        term2= np.dot(self.X.T,itermediate)
        if det(term1)==0:
            raise ValueError("Matrix is singular")
        else:
            return np.dot(inv(term1), term2)


    def gradient_alpha_divergence_loss(self,beta,tau):
        '''
        Finalement, je calcule le gradient de la divergence alpha par rapport à beta.
        J'utilise cette foncition pour la méthode de Nesterov.
        '''
        weight= self.alpha_divergence_weight(beta,tau)
        term1=np.dot((1/self.n) *((self.X.T) * weight.reshape(1, -1)),self.X)
        itermediate=self.y*weight
        term2= np.dot((1/self.n)*self.X.T,itermediate)
        return np.dot(term1,beta)-term2



def score(X,y,beta,tau):
    alp_tool=alpha_divergence_tools(X=X,y=y)
    f_beta=alp_tool.alpha_divergence_loss(beta=beta,tau=tau)
    return f_beta*tau


def calculate_residuals(y_pred,true_y):
        residuals = y_pred - true_y
        return residuals



def grid_search_cv_tukey(
    X: np.ndarray,
    y: np.ndarray,
    c_values: np.ndarray,
    n_splits: int = 5,
) -> float:
    """
    Select the best Tukey constant c via K-Fold cross-validation (median MSE).
 
    Args:
        X: design matrix.
        y: response vector.
        c_values: candidate values for the Tukey constant.
        n_splits: number of CV folds.
 
    Returns:
        best_c: the c value with the lowest median validation error.
    """
    kf = KFold(n_splits=n_splits)
    best_c = None
    best_error = float("inf")
 
    for c in tqdm(c_values, desc="Tukey CV"):
        fold_errors = []
 
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
 
            rlm_model = sm.RLM(y_train, X_train, M=TukeyBiweight(c=c))
            beta_hat = rlm_model.fit().params
 
            y_pred = X_val @ beta_hat
            fold_errors.append(np.mean((y_val - y_pred) ** 2))
 
        median_error = np.median(fold_errors)
        if median_error < best_error:
            best_error = median_error
            best_c = c
 
    return best_c



def grid_search_cv_hampel(
    X: np.ndarray,
    y: np.ndarray,
    a_values: list[float],
    b_values: list[float],
    c_values: list[float],
    n_splits: int = 5,
    random_state: int = None,
) -> dict:
    """
    Select the best Hampel tuning constants (a, b, c) via K-Fold CV (median MSE).
 
    Only evaluates combinations where a < b < c (required by Hampel).
 
    Args:
        X: design matrix.
        y: response vector.
        a_values, b_values, c_values: candidate grids for each constant.
        n_splits: number of CV folds.
        random_state: random seed for fold splits.
 
    Returns:
        best_params: dict with keys 'a', 'b', 'c'.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_params = None
    best_error = float("inf")
 
    for a in a_values:
        for b in b_values:
            for c in c_values:
                if not (a < b < c):
                    continue
 
                fold_errors = []
                for train_idx, val_idx in kf.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
 
                    model = RLM(y_train, X_train, M=Hampel(a=a, b=b, c=c))
                    y_pred = model.fit().predict(X_val)
                    fold_errors.append(np.mean((y_val - y_pred) ** 2))
 
                median_error = np.median(fold_errors)
                if median_error < best_error:
                    best_error = median_error
                    best_params = {"a": a, "b": b, "c": c}
 
    return best_params

