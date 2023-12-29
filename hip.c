#include <stdio.h>
#include <stdlib.h>
#include <math.h>

    typedef struct vector{
        int dim;
        double* data;
    }vector;

double dot( vector w, vector x);
double activation ( double s);
void correction ( vector* w , vector* bias, double des_x, vector x);
void perceptron ( vector* w , vector* bias, double des_x, vector x);
void Printvector(vector X);
void PrintvectorFile( vector A, char* name);
vector StartVector(int n);
vector suma_vector(vector w, vector x);
vector escalar(vector w, double k);
vector Delta_w (vector w, double des_x, vector x);
vector Delta_bias (vector bias, double des_x, vector x);
vector ReadvectorFile( char* name);



int main(void)
{
    vector w, bias , x;
    double des_x = 0 ;
    double s = 0;
    /* Test de dim 2*/
    w = StartVector( 2 );
    bias = StartVector( 1 );
    x = StartVector( 2 );
    /* El fitxer pesos.dat ha de ser de la forma dim valor1 valor 2, recomanat: 2 1 2*/
    w = ReadvectorFile ( "pesos.dat");
    /* El fitxer bias.dat ha de ser de la forma dim valor1 , recomanat: 1 -1*/
    bias = ReadvectorFile ("bias.dat");
    for (int i = 0; i < 10000; i++)
    {
        /* Training data*/
        x.data[0] = cos(1.0*i/10000.0 * 2 * 3.14151692)*0.03;
        x.data[1] = sin(1.0*i/10000.0 * 2 * 3.14151692)*0.03;
        des_x = -1;
        if (x.data[0] + x.data[1] >= 0)
        {
            des_x = 1;
        }
        /* Codi que si que el fa, es una copia de perceptron
        for (int j = 0; j < pow(10,5); j++)
        {
            s = dot(w,x) + (bias).data[0];
            s = activation(s);
            if(abs(s-des_x) <= 0.1 ){
                break;
            }
            correction(&w,&bias,des_x,x); 
        }*/
        /* Codi que no el fa*/
         perceptron(&w,&bias,des_x,x);

    }
    PrintvectorFile(w, "pesos.dat");
    PrintvectorFile(bias, "bias.dat");

    return 0;
}
vector StartVector(int n){
  vector X;
  X.data = (double*)calloc(n,sizeof(double));
  X.dim = n;
  return X;
}

void Printvector(vector X){
  int i;
  printf("\n");
  for(i=0;i<X.dim;i++){
    printf("%lf\t",X.data[i]);
    printf("\n");
  }
  return;
}

double activation ( double s){
    s =-1. +2./(1.+ exp(-s));
    return s;
}
/* Codi que NO  actualitza w ni bias (ignorar bucle infinit)*/
void perceptron ( vector* w , vector* bias, double des_x, vector x ){
    double s = 0;
    s = dot(*w,x) + bias->data[0];
    s = activation(s);
    if(abs(s-des_x) <= 0.1 ){
        return;
    }
    correction(w,bias,des_x,x); 
    perceptron(w,bias,des_x,x);
    return;
}
/* Si que actualitza w i bias*/
void correction ( vector* w , vector* bias, double des_x, vector x){
    int i;
    for ( i = 0; i < (*w).dim; i++){
        (*w).data[i] += Delta_w((*w),des_x,x).data[i];
    }
    for ( i = 0; i < (*bias).dim; i++){
       (*bias).data[i] += Delta_bias(*w,des_x,x).data[i];
    }
    return;
}

/*Producte escalar*/
double dot(vector w, vector x){
    double sum = 0;
    int i = 0;
    int n = w.dim;
    for ( i = 0; i < n; i++)
    {
        sum += w.data[i] * x.data[i];
    }
    return sum;
}
/* suma de vectors, retorna una copia*/
vector suma_vector(vector w, vector x){
    int i = 0;
    int n = w.dim;
    vector v;
    v = StartVector(n);
    for ( i = 0; i < n; i++)
    {
        v.data[i] = w.data[i] + x.data[i];
    }
    return v;
}
/* producte per escalar retorna copia*/
vector escalar(vector w, double k){
    int i = 0;
    int n = w.dim;
    vector v;
    v = StartVector(n);
    for ( i = 0; i < n; i++)
    {
        v.data[i] = w.data[i] * k;
    }
    return v;
}

/*Operador delta per w*/
vector Delta_w (vector w, double des_x, vector x){

    int i = 0, n = w.dim;
    vector v;
    v = StartVector(n);
    for ( i = 0; i < n; i++)
    {
        v.data[i] = des_x * x.data[i];
    }
    return v;
}

/*Operador delta per bias*/
vector Delta_bias (vector bias, double des_x, vector x){

    int i = 0, n = bias.dim;
    vector v;
    v = StartVector(n);
    for ( i = 0; i < n; i++)
    {
        v.data[i] = des_x ;
    }
    return v;
}
/*Llegir vector d'un fitxer*/
vector ReadvectorFile( char* name){
  FILE *fp;
  int i;
  vector C;
  fp = fopen(name,"r");
  if(fp== NULL) return C;
  // LEE la dimension, modificar por scanf si el usuario la introduce manualmente
  fscanf(fp,"%d",&C.dim);
  C = StartVector(C.dim);
  for(i=0;i<C.dim;i++){
      fscanf(fp,"%lf",&C.data[i]);
  }
  fclose (fp);
  return C;
}
/*Imprimir vector a un fitxer*/
void PrintvectorFile( vector A, char* name){
  FILE *fp;
  int i;
  fp = fopen(name,"w");
  if(fp== NULL) return;
  fprintf(fp,"%d\n",A.dim);
  for(i=0;i<A.dim;i++){
      fprintf(fp,"%.2lf\t",A.data[i]);
  }
  fclose (fp);
  return;
}