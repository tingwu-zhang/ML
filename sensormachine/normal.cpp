/*************************************************************************
    > File Name: normal.cpp
    > Author:zhangtx
    > Mail: 18510665908@163.com 
    > Created Time: 2018年01月05日 星期五 10时50分48秒
 ************************************************************************/
#include<iostream>
using namespace std;

int t[][3]={{3,3,1},{4,3,1},{1,1,-1}};//xi1,xi2,yi
int w[3]={0,0,0};//w0,w1,b
int e=1;//learning rate

double compute(int weight[],int test[],int count)
{
    double result=0.0;
    for (int idx=0;idx<count-1;idx++)
    {
        result=result+weight[idx]*test[idx];
    }
    result+=weight[count-1];
    result*=test[count-1];
    return result;
}



int main(int argc,char *argv[])
{
    int iterCount=0;
    while(true)
    {
        bool over=true;
        cout<<endl;

        for(int idx=0;idx<3;idx++)
        {
           if (0>=compute(w,t[idx],3))
           {
               over=false;
               //gradient decent
               w[0]=w[0]+t[idx][0]*t[idx][2];
               w[1]=w[1]+t[idx][1]*t[idx][2];
               w[2]=w[2]+t[idx][2];
               cout<<iterCount++<<" "<<"X"<<idx<< "   "<<w[0]<<" " <<w[1]<<" "<<w[2]<<endl;
           }
           else
           {
               continue;
           }
        }
        if(over==true)
            break;
    }

    cout<<"the moduel is "<<w[0]<<"*x0+"<<w[1]<<"*x1+("<<w[2]<<")"<<endl;


    return 0;
}
