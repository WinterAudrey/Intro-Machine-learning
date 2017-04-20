
#include "jpg.h"
#include "mnist.h"
#include <limits.h>

float linear_classifier(float*w,float*x){
	float d=0;
	for(int i=0; i<784; i++){
		d += w[i]*x[i];
	}
	if(d>=0) return 1;
	else return -1;
}


int main()
{
	//les images d'apprentissage
    float** images = read_mnist("train-images.idx3-ubyte");
	//les labels qui vont avec
	float* labels = read_labels("train-labels.idx1-ubyte");
	//les 10000 images dont il faut deviner le label
	float** test_images = read_mnist("t10k-images.idx3-ubyte");
	//les labels des 10000 images pour tester
	float* test_labels = read_labels("t10k-labels.idx1-ubyte");
	float*w= new float[784];

	//step 1 : initialization
	for(int i=0; i<784; i++){w[i]=(float)rand()*2/INT_MAX-1;}
	float gamma = 0.01;

	//step 2 : learning (que les données d'apprentissage)
	for(int i=0; i<60000; i++){
		printf("%u\n",i);
		//calcul gradient ( g=y*x si erreur, 0 sinon)
		int prediction = linear_classifier(w,images[i]);
		int verite = (labels[i] == 1) ? 1 : -1;
		if(verite!=prediction) {
			printf("ERREUR\n");
			// w(t+1) = w(t) - gamma*y*x
			for(int j=0; j<784; j++) {
				w[j] = w[j] + gamma*verite*images[i][j];
			}
		}
	}


	//step 3 : test (que les données de test)
	float E=0;

    for(int i=0; i<10000; i++) {
		printf("%u\n",i);
		float mind=-1;
		int inference=linear_classifier(w, test_images[i]);
		// inference contient maintenant le label associé à l'image connue
		
      	save_jpg(test_images[i], 28, 28, "%u/%u.jpg", inference, i);
	
		if((inference==1 && test_labels[i]!=1) || (inference==-1 && test_labels[i]==1)){E++; }
		printf("Erreur= %0.2f\n",E*100/i);
    }
    return 0;
}
