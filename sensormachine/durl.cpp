/*************************************************************************
    > File Name: durl.cpp
    > Author:zhangtx
    > Mail: 18510665908@163.com 
    > Created Time: 2018年01月08日 星期一 10时04分10秒
 ************************************************************************/
#include <iomanip>
#include<iostream>
using namespace std;
int a[3]={0,0,0};
int b=0;
int ep=1;
int samples[3][3]={
    {
        3,3,0
    },
    {
        4,3,0
    },
    {
        1,1,0
    }

};
int y[3]={1,1,-1};
int TransSamples[3][3];
int gram[3][3];
void TransposeMatrix(int matrixSrc[][3],int matrixDest[][3],int row,int col)
{
    for(int idxRow=0;idxRow<row;idxRow++)
        for(int idxCol=0;idxCol<col;idxCol++)
            matrixDest[idxCol][idxRow]=matrixSrc[idxRow][idxCol];
}

void MultiMatrix(int matrixLeft[][3],int leftRow,int leftCol,int matrixRight[][3],int rightRow,int rightCol,int result[][3])
{
    for(int idxLeftRow=0;idxLeftRow<leftRow;idxLeftRow++)
    {
        for(int idxRightCol=0;idxRightCol<rightCol;idxRightCol++)
        {

            int sum=0;
            for(int idxRightRow=0;idxRightRow<rightRow;idxRightRow++)
            {
                sum+=matrixLeft[idxLeftRow][idxRightRow]*matrixRight[idxRightRow][idxRightCol];
            }
            result[idxLeftRow][idxRightCol]=sum;
        }
    }
}
void PrintMatrix(int matrix[][3],int row,int col,char *info)
{
    cout<<endl<<info<<endl;
    for(int idxRow=0;idxRow<row;idxRow++)
    {
        cout<<endl;
        for(int idxCol=0;idxCol<col;idxCol++)
        {
            cout<<setw(8)<<matrix[idxRow][idxCol];
        }
    }
    cout<<endl;
}

int w[2]={
    0,0
};

int main(int argc,char *argv[])
{
    PrintMatrix(samples,3,3,"samples");
    TransposeMatrix(samples,TransSamples,3,3);
    PrintMatrix(TransSamples,3,3,"Transposes");
    MultiMatrix(samples,3,3,TransSamples,3,3,gram);
    PrintMatrix(gram,3,3,"result");

    /*train*/
    int iterator=0;
    while(true)
    {
        bool over=true;

        iterator++;
        for(int i=0;i<3;i++)
        {
            int val=0;
            for(int j=0;j<3;j++)
            {
                val+=a[j]*y[j]*gram[j][i];
            }
            
            if (y[i]*(val+b)<=0)
            {
                cout<<" iterator " <<iterator<<"a:";
                for(int k=0;k<3;k++)
                {
                    cout<<setw(8)<<a[k];
                }
                cout<<"   b"<<setw(10)<<b<<endl;
                a[i]=a[i]+ep;
                b=b+y[i];
                over=false;
            }
        }

        if (over==true)
        {

            cout<<" iterator " <<iterator<<"a:";
            for(int i=0;i<3;i++)
            {
                cout<<setw(8)<<a[i];
            }
            cout<<"   b"<<setw(10)<<b<<endl;
            break;
        }
    }

    for(int i=0;i<2;i++)
    {
        for(int j=0;j<3;j++)
            w[i]+=a[j]*y[j]*samples[j][i];
    }
    cout<<endl<<"w="<<w[0]<<"  "<<w[1]<<endl;

    cout<<"the func is "<<w[0]<<"*x1"<<"+"<<w[1]<<"*x2"<<"+("<<b<<")"<<endl;
    return 0;
}
