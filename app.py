import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import mlxtend
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

Ushape=pd.read_csv(r"C:\Users\HP\Downloads\Multiple CSV\Multiple CSV\1.ushape.csv",header=None,names=["f1","f2","cv"])
Concentriccir1=pd.read_csv(r"C:\Users\HP\Downloads\Multiple CSV\Multiple CSV\2.concerticcir1.csv",header=None,names=["f1","f2","cv"])
Concentriccir2=pd.read_csv(r"C:\Users\HP\Downloads\Multiple CSV\Multiple CSV\3.concertriccir2.csv",header=None,names=["f1","f2","cv"])
Linearsep=pd.read_csv(r"C:\Users\HP\Downloads\Multiple CSV\Multiple CSV\4.linearsep.csv",header=None,names=["f1","f2","cv"])
Outlier=pd.read_csv(r"C:\Users\HP\Downloads\Multiple CSV\Multiple CSV\5.outlier.csv",header=None,names=["f1","f2","cv"])
Overlap=pd.read_csv(r"C:\Users\HP\Downloads\Multiple CSV\Multiple CSV\6.overlap.csv",header=None,names=["f1","f2","cv"])
Xor=pd.read_csv(r"C:\Users\HP\Downloads\Multiple CSV\Multiple CSV\7.xor.csv",header=None,names=["f1","f2","cv"])
Twospirals=pd.read_csv(r"C:\Users\HP\Downloads\Multiple CSV\Multiple CSV\8.twospirals.csv",header=None,names=["f1","f2","cv"])
Random=pd.read_csv(r"C:\Users\HP\Downloads\Multiple CSV\Multiple CSV\9.random.csv",header=None,names=["f1","f2","cv"])




with st.sidebar:
    st.title("Side Quest")
    radio_button=st.selectbox("Datasets",["Ushape","Concentriccir1","Concentriccir2","Linearsep","Outlier","Overlap","Xor","Twospirals","Random"])
    k=st.number_input("K-value",min_value=1,max_value=20)
    radio_button2=st.selectbox("Decision_surfaces",["Single","Multiple"])

if radio_button=="Ushape":
    st.write("Ushape")
    st.scatter_chart(data=Ushape,x="f1",y="f2")
    fv=Ushape.iloc[:,:2]
    cv=Ushape.iloc[:,-1]
    std=StandardScaler()
    train_fv=std.fit_transform(fv) 
    if(radio_button2=="Single"):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_fv,cv.astype(int))
        fig, ax=plt.subplots()
        plot_decision_regions(X=train_fv,y=cv.astype(int).values,clf=knn,ax=ax)
        st.pyplot(fig)
    elif(radio_button2=="Multiple"):
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        for i, ax in zip(range(1, k, 2), axes.flatten()):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(train_fv, cv.astype(int))
            plot_decision_regions(X=train_fv, y=cv.astype(int).values, clf=knn, ax=ax)
            ax.set_title(f'KNN with n_neighbors={i}')
        plt.tight_layout()
        st.pyplot(fig)
    x_train,x_test,y_train,y_test=train_test_split(fv,cv,train_size=0.8,random_state=10,stratify=cv)
    std=StandardScaler()
    stand_x_train=std.fit_transform(x_train)
    stand_x_test=std.transform(x_test)
    train_err=[]
    test_err=[]
    for n in range(1,k,2):
        knn=KNeighborsClassifier(n_neighbors=n)
        learned_knn=knn.fit(stand_x_train,y_train)
        tr_predicted=learned_knn.predict(stand_x_train)
        train_err.append(1-accuracy_score(y_train,tr_predicted))
        tst_predicted=learned_knn.predict(stand_x_test)
        test_err.append(1-accuracy_score(y_test,tst_predicted))
    error={'Train':train_err,'Test':test_err}
    error_df=pd.DataFrame(error)
    st.line_chart(error_df)
    
        







if radio_button=="Concentriccir1":
    st.write("Concentriccir1")
    st.scatter_chart(data=Concentriccir1,x="f1",y="f2")
    fv=Concentriccir1.iloc[:,:2]
    cv=Concentriccir1.iloc[:,-1]
    std=StandardScaler()
    train_fv=std.fit_transform(fv)
    if(radio_button2=="Single"):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_fv,cv.astype(int))
        fig, ax=plt.subplots()
        plot_decision_regions(X=train_fv,y=cv.astype(int).values,clf=knn,ax=ax)
        st.pyplot(fig)
    elif(radio_button2=="Multiple"):
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        for i, ax in zip(range(1, k, 2), axes.flatten()):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(train_fv, cv.astype(int))
            plot_decision_regions(X=train_fv, y=cv.astype(int).values, clf=knn, ax=ax)
            ax.set_title(f'KNN with n_neighbors={i}')
        plt.tight_layout()
        st.pyplot(fig)
    x_train,x_test,y_train,y_test=train_test_split(fv,cv,train_size=0.8,random_state=10,stratify=cv)
    std=StandardScaler()
    stand_x_train=std.fit_transform(x_train)
    stand_x_test=std.transform(x_test)
    train_err=[]
    test_err=[]
    for n in range(1,k,2):
        knn=KNeighborsClassifier(n_neighbors=n)
        learned_knn=knn.fit(stand_x_train,y_train)
        tr_predicted=learned_knn.predict(stand_x_train)
        train_err.append(1-accuracy_score(y_train,tr_predicted))
        tst_predicted=learned_knn.predict(stand_x_test)
        test_err.append(1-accuracy_score(y_test,tst_predicted))
    error={'Train':train_err,'Test':test_err}
    error_df=pd.DataFrame(error)
    st.line_chart(error_df) 
 
    

if radio_button=="Concentriccir2":
    st.write("Concentriccir2")
    st.scatter_chart(data=Concentriccir2,x="f1",y="f2")
    fv=Concentriccir2.iloc[:,:2]
    cv=Concentriccir2.iloc[:,-1]
    std=StandardScaler()
    train_fv=std.fit_transform(fv) 
    if(radio_button2=="Single"):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_fv,cv.astype(int))
        fig, ax=plt.subplots()
        plot_decision_regions(X=train_fv,y=cv.astype(int).values,clf=knn,ax=ax)
        st.pyplot(fig)
    elif(radio_button2=="Multiple"):
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        for i, ax in zip(range(1, k, 2), axes.flatten()):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(train_fv, cv.astype(int))
            plot_decision_regions(X=train_fv, y=cv.astype(int).values, clf=knn, ax=ax)
            ax.set_title(f'KNN with n_neighbors={i}')
        plt.tight_layout()
        st.pyplot(fig)
    x_train,x_test,y_train,y_test=train_test_split(fv,cv,train_size=0.8,random_state=10,stratify=cv)
    std=StandardScaler()
    stand_x_train=std.fit_transform(x_train)
    stand_x_test=std.transform(x_test)
    train_err=[]
    test_err=[]
    for n in range(1,k,2):
        knn=KNeighborsClassifier(n_neighbors=n)
        learned_knn=knn.fit(stand_x_train,y_train)
        tr_predicted=learned_knn.predict(stand_x_train)
        train_err.append(1-accuracy_score(y_train,tr_predicted))
        tst_predicted=learned_knn.predict(stand_x_test)
        test_err.append(1-accuracy_score(y_test,tst_predicted))
    error={'Train':train_err,'Test':test_err}
    error_df=pd.DataFrame(error)
    st.line_chart(error_df)


if radio_button=="Linearsep":
    st.write("Linearsep")
    st.scatter_chart(data=Linearsep,x="f1",y="f2")
    fv=Linearsep.iloc[:,:2]
    cv=Linearsep.iloc[:,-1]
    std=StandardScaler()
    train_fv=std.fit_transform(fv) 
    if(radio_button2=="Single"):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_fv,cv.astype(int))
        fig, ax=plt.subplots()
        plot_decision_regions(X=train_fv,y=cv.astype(int).values,clf=knn,ax=ax)
        st.pyplot(fig)
    elif(radio_button2=="Multiple"):
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        for i, ax in zip(range(1, k, 2), axes.flatten()):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(train_fv, cv.astype(int))
            plot_decision_regions(X=train_fv, y=cv.astype(int).values, clf=knn, ax=ax)
            ax.set_title(f'KNN with n_neighbors={i}')
        plt.tight_layout()
        st.pyplot(fig)
    x_train,x_test,y_train,y_test=train_test_split(fv,cv,train_size=0.8,random_state=10,stratify=cv)
    std=StandardScaler()
    stand_x_train=std.fit_transform(x_train)
    stand_x_test=std.transform(x_test)
    train_err=[]
    test_err=[]
    for n in range(1,k,2):
        knn=KNeighborsClassifier(n_neighbors=n)
        learned_knn=knn.fit(stand_x_train,y_train)
        tr_predicted=learned_knn.predict(stand_x_train)
        train_err.append(1-accuracy_score(y_train,tr_predicted))
        tst_predicted=learned_knn.predict(stand_x_test)
        test_err.append(1-accuracy_score(y_test,tst_predicted))
    error={'Train':train_err,'Test':test_err}
    error_df=pd.DataFrame(error)
    st.line_chart(error_df)


if radio_button=="Outlier":
    st.write("Outlier")
    st.scatter_chart(data=Outlier,x="f1",y="f2")
    fv=Outlier.iloc[:,:2]
    cv=Outlier.iloc[:,-1]
    std=StandardScaler()
    train_fv=std.fit_transform(fv) 
    if(radio_button2=="Single"):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_fv,cv.astype(int))
        fig, ax=plt.subplots()
        plot_decision_regions(X=train_fv,y=cv.astype(int).values,clf=knn,ax=ax)
        st.pyplot(fig)
    elif(radio_button2=="Multiple"):
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        for i, ax in zip(range(1, k, 2), axes.flatten()):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(train_fv, cv.astype(int))
            plot_decision_regions(X=train_fv, y=cv.astype(int).values, clf=knn, ax=ax)
            ax.set_title(f'KNN with n_neighbors={i}')
        plt.tight_layout()
        st.pyplot(fig)
    x_train,x_test,y_train,y_test=train_test_split(fv,cv,train_size=0.8,random_state=10,stratify=cv)
    std=StandardScaler()
    stand_x_train=std.fit_transform(x_train)
    stand_x_test=std.transform(x_test)
    train_err=[]
    test_err=[]
    for n in range(1,k,2):
        knn=KNeighborsClassifier(n_neighbors=n)
        learned_knn=knn.fit(stand_x_train,y_train)
        tr_predicted=learned_knn.predict(stand_x_train)
        train_err.append(1-accuracy_score(y_train,tr_predicted))
        tst_predicted=learned_knn.predict(stand_x_test)
        test_err.append(1-accuracy_score(y_test,tst_predicted))
    error={'Train':train_err,'Test':test_err}
    error_df=pd.DataFrame(error)
    st.line_chart(error_df)

if radio_button=="Overlap":
    st.write("Overlap")
    st.scatter_chart(data=Overlap,x="f1",y="f2")
    fv=Overlap.iloc[:,:2]
    cv=Overlap.iloc[:,-1]
    std=StandardScaler()
    train_fv=std.fit_transform(fv) 
    if(radio_button2=="Single"):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_fv,cv.astype(int))
        fig, ax=plt.subplots()
        plot_decision_regions(X=train_fv,y=cv.astype(int).values,clf=knn,ax=ax)
        st.pyplot(fig)
    elif(radio_button2=="Multiple"):
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        for i, ax in zip(range(1, k, 2), axes.flatten()):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(train_fv, cv.astype(int))
            plot_decision_regions(X=train_fv, y=cv.astype(int).values, clf=knn, ax=ax)
            ax.set_title(f'KNN with n_neighbors={i}')
        plt.tight_layout()
        st.pyplot(fig)
    x_train,x_test,y_train,y_test=train_test_split(fv,cv,train_size=0.8,random_state=10,stratify=cv)
    std=StandardScaler()
    stand_x_train=std.fit_transform(x_train)
    stand_x_test=std.transform(x_test)
    train_err=[]
    test_err=[]
    for n in range(1,k,2):
        knn=KNeighborsClassifier(n_neighbors=n)
        learned_knn=knn.fit(stand_x_train,y_train)
        tr_predicted=learned_knn.predict(stand_x_train)
        train_err.append(1-accuracy_score(y_train,tr_predicted))
        tst_predicted=learned_knn.predict(stand_x_test)
        test_err.append(1-accuracy_score(y_test,tst_predicted))
    error={'Train':train_err,'Test':test_err}
    error_df=pd.DataFrame(error)
    st.line_chart(error_df)


if radio_button=="Xor":
    st.write("Xor")
    st.scatter_chart(data=Xor,x="f1",y="f2")
    fv=Xor.iloc[:,:2]
    cv=Xor.iloc[:,-1]
    std=StandardScaler()
    train_fv=std.fit_transform(fv) 
    if(radio_button2=="Single"):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_fv,cv.astype(int))
        fig, ax=plt.subplots()
        plot_decision_regions(X=train_fv,y=cv.astype(int).values,clf=knn,ax=ax)
        st.pyplot(fig)
    elif(radio_button2=="Multiple"):
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        for i, ax in zip(range(1, k, 2), axes.flatten()):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(train_fv, cv.astype(int))
            plot_decision_regions(X=train_fv, y=cv.astype(int).values, clf=knn, ax=ax)
            ax.set_title(f'KNN with n_neighbors={i}')
        plt.tight_layout()
        st.pyplot(fig)
    x_train,x_test,y_train,y_test=train_test_split(fv,cv,train_size=0.8,random_state=10,stratify=cv)
    std=StandardScaler()
    stand_x_train=std.fit_transform(x_train)
    stand_x_test=std.transform(x_test)
    train_err=[]
    test_err=[]
    for n in range(1,k,2):
        knn=KNeighborsClassifier(n_neighbors=n)
        learned_knn=knn.fit(stand_x_train,y_train)
        tr_predicted=learned_knn.predict(stand_x_train)
        train_err.append(1-accuracy_score(y_train,tr_predicted))
        tst_predicted=learned_knn.predict(stand_x_test)
        test_err.append(1-accuracy_score(y_test,tst_predicted))
    error={'Train':train_err,'Test':test_err}
    error_df=pd.DataFrame(error)
    st.line_chart(error_df)

if radio_button=="Twospirals":
    st.write("Twospirals")
    st.scatter_chart(data=Twospirals,x="f1",y="f2")
    fv=Twospirals.iloc[:,:2]
    cv=Twospirals.iloc[:,-1]
    std=StandardScaler()
    train_fv=std.fit_transform(fv) 
    if(radio_button2=="Single"):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_fv,cv.astype(int))
        fig, ax=plt.subplots()
        plot_decision_regions(X=train_fv,y=cv.astype(int).values,clf=knn,ax=ax)
        st.pyplot(fig)
    elif(radio_button2=="Multiple"):
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        for i, ax in zip(range(1, k, 2), axes.flatten()):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(train_fv, cv.astype(int))
            plot_decision_regions(X=train_fv, y=cv.astype(int).values, clf=knn, ax=ax)
            ax.set_title(f'KNN with n_neighbors={i}')
        plt.tight_layout()
        st.pyplot(fig)
    x_train,x_test,y_train,y_test=train_test_split(fv,cv,train_size=0.8,random_state=10,stratify=cv)
    std=StandardScaler()
    stand_x_train=std.fit_transform(x_train)
    stand_x_test=std.transform(x_test)
    train_err=[]
    test_err=[]
    for n in range(1,k,2):
        knn=KNeighborsClassifier(n_neighbors=n)
        learned_knn=knn.fit(stand_x_train,y_train)
        tr_predicted=learned_knn.predict(stand_x_train)
        train_err.append(1-accuracy_score(y_train,tr_predicted))
        tst_predicted=learned_knn.predict(stand_x_test)
        test_err.append(1-accuracy_score(y_test,tst_predicted))
    error={'Train':train_err,'Test':test_err}
    error_df=pd.DataFrame(error)
    st.line_chart(error_df)


if radio_button=="Random":
    st.write("Random")
    st.scatter_chart(data=Random,x="f1",y="f2")
    fv=Random.iloc[:,:2]
    cv=Random.iloc[:,-1]
    std=StandardScaler()
    train_fv=std.fit_transform(fv) 
    if(radio_button2=="Single"):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_fv,cv.astype(int))
        fig, ax=plt.subplots()
        plot_decision_regions(X=train_fv,y=cv.astype(int).values,clf=knn,ax=ax)
        st.pyplot(fig)
    elif(radio_button2=="Multiple"):
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        for i, ax in zip(range(1, k, 2), axes.flatten()):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(train_fv, cv.astype(int))
            plot_decision_regions(X=train_fv, y=cv.astype(int).values, clf=knn, ax=ax)
            ax.set_title(f'KNN with n_neighbors={i}')
        plt.tight_layout()
        st.pyplot(fig)
    x_train,x_test,y_train,y_test=train_test_split(fv,cv,train_size=0.8,random_state=10,stratify=cv)
    std=StandardScaler()
    stand_x_train=std.fit_transform(x_train)
    stand_x_test=std.transform(x_test)
    train_err=[]
    test_err=[]
    for n in range(1,k,2):
        knn=KNeighborsClassifier(n_neighbors=n)
        learned_knn=knn.fit(stand_x_train,y_train)
        tr_predicted=learned_knn.predict(stand_x_train)
        train_err.append(1-accuracy_score(y_train,tr_predicted))
        tst_predicted=learned_knn.predict(stand_x_test)
        test_err.append(1-accuracy_score(y_test,tst_predicted))
    error={'Train':train_err,'Test':test_err}
    error_df=pd.DataFrame(error)
    st.line_chart(error_df)




