/**************************************************************
* Copyright 2016 Aleksandr Voishchev
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http ://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*************************************************************/

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

class NeuralNet
{
public:

        long m_lWeightsCount;
        long m_lNodesCount;
        long m_lLayersCount;

		double * m_pdWeights;
        double * m_pdNodes; 
		long * m_plLayers; 

        long m_lPatternsCount;
        double * m_pdPatterns;

        double GetRelevance(long lNode);
//        CString GetNMExpression(long lNode);

private:

		long * m_plNodePtr;
        long * m_plWeightPtr;
        double * m_pdWeightsOld;
        double * m_pdGradient;
        double * m_pdOldGradient;
        double * m_pdDelta;
        double * m_pdOldDelta;
        double * m_pdDerivatives;
        double * m_pdErrors;

public:
        NeuralNet();
        virtual ~NeuralNet();

        void createPatterns(long lPatterns);
        void setPattern(long lPattern, long lUnit, double dValue);

		int createNetwork(long nLayersCount, long *nLayers);
        void initializeNetwork(unsigned int uiRandomSeed);

        void trainBPROP(long lLength, double dMaxError, double dLearningRate=0.5, double dMomentum=0.1);
        void trainRPROP(long lLength, double dMaxError, double dIncreaseFactor=1.2, double dDecreaseFactor=0.5, double dDeltaMin=1e-6, double dDeltaMax=50.0, double dDeltaInit=0.1);
        void trainSCG  (long lLength, double dMaxError, double dSigma=1e-4, double dLambda=1e-6);
		void testPattern(long lPattern);

		double getError();
		double getErrorWithoutUnit(long lNumUnit);

        double m_dTolerance; 

protected:
		void clearNetwork();
        void freeNetworkMemory();

private:
		inline double sign(double a) {if (a > 0) return 1.0; if (a < 0) return -1.0; return 0.0;};
		inline double sign1(double a) {if (a >= 0) return 1.0; return -1.0;};
		inline double product_of_xt_by_y(double *x, double *y, int tab_size){ double sum = 0 ; for (long indice = 0 ; indice < tab_size ; indice++) sum += x[indice]*y[indice] ; return(sum);}
		inline double square_of_norm(double *x, int tab_size){return(product_of_xt_by_y(x,x,tab_size));}
        inline double backPropagate(double *pWeights);
};

