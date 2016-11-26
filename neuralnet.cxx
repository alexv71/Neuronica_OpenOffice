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

#include "NeuralNet.h"
#include <stdio.h>  
#include <stdlib.h> 
#include <math.h>
#include <float.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

NeuralNet::NeuralNet()
{
        clearNetwork();
}

NeuralNet::~NeuralNet()
{
        freeNetworkMemory();
}

void NeuralNet::clearNetwork()
{
        m_lLayersCount = 0;
        m_lNodesCount = 0;
        m_lWeightsCount = 0;
        m_lPatternsCount = 0;
        m_lPatternsCount = 0;

        m_plLayers = NULL;
        m_pdNodes = NULL;
        m_pdWeights = NULL;
        m_pdWeightsOld = NULL;
        m_pdErrors = NULL;
        m_pdGradient = NULL;
        m_plNodePtr = NULL;
        m_plWeightPtr = NULL;
        m_pdPatterns = NULL;
        m_pdPatterns = NULL;

        m_pdOldGradient = NULL;
        m_pdDelta = NULL;
        m_pdOldDelta = NULL;

}

void NeuralNet::freeNetworkMemory()
{
        delete [] m_plLayers;
        delete [] m_pdNodes;

        delete [] m_pdWeights;
        delete [] m_pdWeightsOld;

        delete [] m_pdErrors;
        delete [] m_pdGradient;
        delete [] m_plNodePtr;
        delete [] m_plWeightPtr;
        delete [] m_pdPatterns;
        delete [] m_pdOldGradient;
        delete [] m_pdDelta;
        delete [] m_pdOldDelta;
}

int NeuralNet::createNetwork(long nLayersCount, long *nLayers)
{
        m_lLayersCount = nLayersCount;

        freeNetworkMemory();
        m_pdPatterns = NULL;
        m_lPatternsCount = 0;
        m_pdPatterns = NULL;
        m_lPatternsCount = 0;

        m_plLayers = new long[m_lLayersCount];
        m_plWeightPtr = new long[m_lLayersCount];
        m_plNodePtr = new long[m_lLayersCount];

        int i;

        m_lNodesCount = 0;
        m_lWeightsCount = 0;

        for (i = 0; i < m_lLayersCount; i++)
        {
              m_plLayers[i] = nLayers[i];
              m_lNodesCount += m_plLayers[i];
        }

        m_plWeightPtr[0] = 0;
        m_plNodePtr[0] = 0;
        for (i = 1; i < m_lLayersCount; i++)
        {
//                m_lWeightsCount += m_plLayers[i]*(m_plLayers[i-1]);
                m_lWeightsCount += m_plLayers[i]*(m_plLayers[i-1]+1);
//!!!!!!
                m_plWeightPtr[i] = m_lWeightsCount;
                m_plNodePtr[i] = m_plNodePtr[i-1]+m_plLayers[i-1];
        }

        m_pdNodes = new double[m_lNodesCount];

        m_pdWeights = new double[m_lWeightsCount];
        m_pdWeightsOld = new double[m_lWeightsCount];

        m_pdGradient = new double[m_lWeightsCount];
        m_pdErrors = new double[m_lNodesCount];
        m_pdOldDelta = new double[m_lWeightsCount];
        m_pdOldGradient = new double[m_lWeightsCount];
        m_pdDelta = new double[m_lWeightsCount];

	    for (i = 0; i < m_lWeightsCount; i++)
                m_pdWeights[i] = 0;

        for (i = 0; i < m_lNodesCount; i++)
              m_pdNodes[i] = 0;

	return 0;
}

void NeuralNet::initializeNetwork(unsigned int uiRandomSeed)
{
	long i;	
	srand(uiRandomSeed);
        for (i = 0; i < m_lWeightsCount; i++)
        {
                m_pdWeights[i] = (0.5-(rand() % 1000)/1000.0)*2;
                m_pdWeightsOld[i] = m_pdWeights[i];
        }
// переписать нормально на с++
        for (i = 0; i < m_lNodesCount; i++)
//              m_pdNodes[i] = 0;
              m_pdNodes[i] = (rand() % 1000)/1000.0;

        return;
}

inline double NeuralNet::backPropagate(double *pWeights){
// Прямой процесс
        long    pStatePrev;     // Указатель на состояние нейрона предыдущего слоя
        long    pWeightPrev;// Указатель на вес из предыдущего слоя
        long    pPattern = 0;      // Указатель на текущее значение паттерна обучения
        long    pErrorPattern;
        long    pState;
        long    pWeight;
        double  Sum;
        double  Error;
        double  ErrorSum = 0;

        long i,j,k,pattern,ep;
        for (ep = 0; ep < m_lWeightsCount; ep++)
                m_pdGradient[ep] = 0.0;
        for (pattern = 0; pattern < m_lPatternsCount; pattern++)
        {
                pState = 0;
                pWeight = 0;
                // Прямой процесс
                // Первый слой
                for (i = 0; i < m_plLayers[0]; i++)
                        m_pdNodes[pState++] = m_pdPatterns[pPattern++];
                // Остальные слои
                for (i = 1; i < m_lLayersCount; i++)
                {
                        for ( j = 0; j < m_plLayers[i]; j++)
                        {
                                Sum = pWeights[pWeight++];
                                pStatePrev = m_plNodePtr[i-1];
                                for (k = 0; k < m_plLayers[i-1]; k++)
                                        Sum += m_pdNodes[pStatePrev++]*pWeights[pWeight++];

                                if (0.0-Sum < DBL_MAX_10_EXP)
                                        m_pdNodes[pState++] = 1.0 / (1 + exp(0.0-Sum));
                                else
                                        m_pdNodes[pState++] = 0;
                        }
                }
                pErrorPattern = pPattern;
                // Обратный процесс
                pState -= 1;
                // Последний слой
                long OutCount = m_plLayers[m_lLayersCount-1];
                for (i = 0; i < OutCount; i++)
                {
                        long OutIndex = m_lNodesCount - OutCount + i;
                        m_pdErrors[OutIndex] = (m_pdNodes[OutIndex] - m_pdPatterns[pPattern++]) * m_pdNodes[OutIndex] * (1 - m_pdNodes[OutIndex]);
//                      m_pdErrors[pState] = (m_pdNodes[pState] - m_pdPatterns[pPattern++]) * m_pdNodes[pState] * (1 - m_pdNodes[pState]);
                        pState--;
                }
                // Остальные слои
                for (i = m_lLayersCount-2; i > 0; i--)
                {
                        for (j = m_plLayers[i] - 1; j >= 0; j--)
                        {
                                Sum = 0;
                                pWeightPrev = m_plWeightPtr[i] + 1 + j;
                                pStatePrev = m_plNodePtr[i + 1];
                                for (k = 0; k < m_plLayers[i+1]; k++)
                                {
                                        Sum += m_pdErrors[pStatePrev++] * pWeights[pWeightPrev];
                                        pWeightPrev += m_plLayers[i] + 1;
                                }
                                m_pdErrors[pState] = Sum * m_pdNodes[pState] * (1 - m_pdNodes[pState]);
                                pState--;
                        }
                }
                pWeight = 0;
                pState = m_plNodePtr[1];
                for (i = 1; i < m_lLayersCount; i++)
                {
                        for (j = 0; j < m_plLayers[i]; j++)
                        {
                                pStatePrev = m_plNodePtr[i-1];
                                m_pdGradient[pWeight++] += m_pdErrors[pState];
                                for (k = 0; k < m_plLayers[i-1]; k++)
                                        m_pdGradient[pWeight++] += m_pdNodes[pStatePrev++]*m_pdErrors[pState];
                                pState++;
                        }
                }
                // Calculate Error
                pState = m_lNodesCount-1;
                for (i = 0; i < m_plLayers[m_lLayersCount-1]; i++)
                {
                        Error = m_pdNodes[pState] - m_pdPatterns[pErrorPattern++];
                        Error *= Error;
                        ErrorSum += Error;
                        pState--;
                }
        }// End of Pattern
        return ErrorSum;
}

void  NeuralNet::setPattern(long lPattern, long lUnit, double dValue){
        m_pdPatterns[(m_plLayers[0] + m_plLayers[m_lLayersCount - 1])*
                lPattern + lUnit] = dValue;
        return;
}

/////////////////////////////////////////////////////////////////////////////
//
void NeuralNet::trainBPROP(long lLength, double dMaxError, double dLearningRate, double dMomentum)
{
        long    pWeight;        // Указатель на текущий вес
        double  dError, dWeightOld;

        long i,epoch,j,k;

// Инициализация
		for (i = 0; i < m_lWeightsCount; i++)
        {
            m_pdOldDelta[i] = 0;
            m_pdGradient[i] = 0.0;
            m_pdOldGradient[i] = 0.0;
        }

// Обучение
        for (epoch = 0; epoch < lLength ; epoch++)
        {
                // BackPropagation
                dError = backPropagate(m_pdWeights);
                if (dError < m_dTolerance) break;
                // Обновление весов
                pWeight = 0;
                for (i = 1; i < m_lLayersCount; i++)
                {
                        for (j = 0; j < m_plLayers[i]; j++)
                        {
					        dWeightOld = m_pdWeights[pWeight];
					        m_pdWeights[pWeight] -= m_pdGradient[pWeight] * dLearningRate + dMomentum * (m_pdWeights[pWeight] - m_pdWeightsOld[pWeight]);
							m_pdWeightsOld[pWeight] = dWeightOld;
//							UpdateBPWeight(pWeight,m_pdGradient[pWeight]);
                            pWeight++;
                            for (k = 0; k < m_plLayers[i-1]; k++)
                            {
								dWeightOld = m_pdWeights[pWeight];
								m_pdWeights[pWeight] -= m_pdGradient[pWeight] * dLearningRate + dMomentum * (m_pdWeights[pWeight] - m_pdWeightsOld[pWeight]);
								m_pdWeightsOld[pWeight] = dWeightOld;
//                              UpdateBPWeight(pWeight,m_pdGradient[pWeight]);
                                pWeight++;
                            }
                        }
                }
        }

        return ;
}


void NeuralNet::trainRPROP(long lLength, double dMaxError, double dIncreaseFactor, double dDecreaseFactor, double dDeltaMin, double dDeltaMax, double dDeltaInit){
//      long    pState;         // Указатель на состояние текущего нейрона
//      long    pStatePrev;     // Указатель на состояние нейрона предыдущего слоя
        long    pWeight;        // Указатель на текущий вес
        double  dError;

        long i,epoch,j,ep,k;
// Инициализация


        for (i = 0; i < m_lWeightsCount; i++)
        {
            m_pdOldDelta[i] = 0;
            m_pdDelta[i] = dDeltaInit;
            m_pdGradient[i] = 0.0;
            m_pdOldGradient[i] = 0.0;
        }

// Обучение
        for (epoch = 0; epoch < lLength ; epoch++)
        {
                for (ep = 0; ep < m_lWeightsCount; ep++)
                        m_pdOldGradient[ep] = m_pdGradient[ep];

                // BackPropagation
                dError = backPropagate(m_pdWeights);
                if (dError < m_dTolerance) break;
                // Обновление весов
                pWeight = 0;
                for (i = 1; i < m_lLayersCount; i++)
                {
                        for (j = 0; j < m_plLayers[i]; j++)
                        {
//                                UpdateRPWeight(pWeight,m_pdGradient[pWeight]);
						        if (m_pdOldGradient[pWeight] * m_pdGradient[pWeight] > 0)
								{
									m_pdDelta[pWeight] = fmin(m_pdDelta[pWeight]*dIncreaseFactor,dDeltaMax);
									m_pdOldDelta[pWeight] = -sign(m_pdGradient[pWeight]) * m_pdDelta[pWeight];
									m_pdWeights[pWeight] += m_pdOldDelta[pWeight];
								}
								else if (m_pdOldGradient[pWeight] * m_pdGradient[pWeight] < 0)
								{
										m_pdDelta[pWeight] = fmax(m_pdDelta[pWeight]*dDecreaseFactor,dDeltaMin);
										m_pdOldDelta[pWeight] = -m_pdOldDelta[pWeight];
										m_pdWeights[pWeight] += m_pdOldDelta[pWeight];
										m_pdGradient[pWeight] = 0;
								}
								else
								{
										m_pdOldDelta[pWeight] = -sign(m_pdGradient[pWeight]) * m_pdDelta[pWeight];
										m_pdWeights[pWeight] += m_pdOldDelta[pWeight];
								}
                                pWeight++;
                                for (k = 0; k < m_plLayers[i-1]; k++)
                                {
//                                        UpdateRPWeight(pWeight,m_pdGradient[pWeight]);
										if (m_pdOldGradient[pWeight] * m_pdGradient[pWeight] > 0)
										{
											m_pdDelta[pWeight] = fmin(m_pdDelta[pWeight]*dIncreaseFactor,dDeltaMax);
											m_pdOldDelta[pWeight] = -sign(m_pdGradient[pWeight]) * m_pdDelta[pWeight];
											m_pdWeights[pWeight] += m_pdOldDelta[pWeight];
										}
										else if (m_pdOldGradient[pWeight] * m_pdGradient[pWeight] < 0)
										{
												m_pdDelta[pWeight] = fmax(m_pdDelta[pWeight]*dDecreaseFactor,dDeltaMin);
												m_pdOldDelta[pWeight] = -m_pdOldDelta[pWeight];
												m_pdWeights[pWeight] += m_pdOldDelta[pWeight];
												m_pdGradient[pWeight] = 0;
										}
										else
										{
												m_pdOldDelta[pWeight] = -sign(m_pdGradient[pWeight]) * m_pdDelta[pWeight];
												m_pdWeights[pWeight] += m_pdOldDelta[pWeight];
										}
                                        pWeight++;
                                }
                        }
                }
		}
        return ;
}

void NeuralNet::trainSCG(long lLength, double dMaxError, double dSigma, double dLambda){
        double *p = new double[m_lWeightsCount];
        double *r = new double[m_lWeightsCount];
        double *step = new double[m_lWeightsCount];

        long i, k, epoch, success ;
	    double delta, norm_of_p_2, lmd, lmd_bar, current_dError, old_dError, sgm, mu, alpha, grand_delta, beta ;

// now we choose initial values for SCG parameters
//              step1:
                lmd = dLambda ;
                lmd_bar = 0.0 ;
                success = true ;
                k = 1 ;
                epoch =0;

                for (i = 0; i < m_lNodesCount; i++)
                        m_pdNodes[i] = 0.0;
                current_dError = backPropagate(m_pdWeights);

                for (i=0 ; i < m_lWeightsCount ; i++)
                {
                        p[i] = - m_pdGradient[i] ;
                        r[i] = p[i] ;
                }

step2:
                if (success)
                {
                        norm_of_p_2 = square_of_norm(p,m_lWeightsCount) ;
                        sgm = dSigma/sqrt(norm_of_p_2) ;
                        for (i=0 ; i < m_lWeightsCount ; i++)
                        {
                                m_pdOldGradient[i] = m_pdGradient[i] ;
                                m_pdOldDelta[i] = m_pdWeights[i];
                        }
                        old_dError = current_dError ;
                        for (i=0 ; i < m_lWeightsCount ; i++)
                                m_pdWeights[i] += sgm*p[i] ;

                        backPropagate(m_pdWeights);
                        for (i=0 ; i < m_lWeightsCount ; i++)
                                step[i] = (m_pdGradient[i]-m_pdOldGradient[i])/sgm ;

                        delta = product_of_xt_by_y(p,step,m_lWeightsCount) ;
                }

                delta += (lmd - lmd_bar) * norm_of_p_2 ;

                if (delta <=0)
                {
                        lmd_bar = 2.0 * (lmd - delta/norm_of_p_2) ;
                        delta = -delta + lmd*norm_of_p_2 ;
                        lmd = lmd_bar ;
                }

                mu = product_of_xt_by_y(p,r,m_lWeightsCount) ;
                alpha = mu/delta ;

                for (i=0 ; i < m_lWeightsCount ; i++)
                        m_pdWeights[i]=m_pdOldDelta[i]+alpha*p[i];

                current_dError = backPropagate(m_pdWeights);

                grand_delta = 2.0*delta*(old_dError-current_dError)/(mu*mu) ;
                if (grand_delta >= 0)
                {
                        double r_sum = 0.0;
                        double tmp ;

                        for (i=0 ; i < m_lWeightsCount ; i++)
                        {
                                tmp = -1.0 * m_pdGradient[i] ;
                                r_sum +=  tmp * r[i] ;
                                r[i] = tmp ;
                        }

                        lmd_bar = 0 ;
                        success = true ;

                        if (k >= m_lWeightsCount)
                        {
                                for (i=0 ; i < m_lWeightsCount ; i++)
                                        p[i] = r[i] ;

                                lmd = dLambda ;
                                lmd_bar = 0.0 ;
                                success = true ;
                                k = 1 ;
                        }
                        else
                        {
                                beta = (square_of_norm(r,m_lWeightsCount) - r_sum)/mu ;
                                for (i=0 ; i < m_lWeightsCount ; i++)
                                        p[i] = r[i]+ beta*p[i] ;
                        }

                        if (grand_delta >=0.75)
                                lmd = lmd/4.0 ;
                }

                else
                {
                        for (i=0 ; i < m_lWeightsCount ; i++)
                                m_pdWeights[i] = m_pdOldDelta[i] ;
                        current_dError = old_dError ;
                        lmd_bar = lmd ;
                        success = false ;
                }

                if (grand_delta < 0.25)
                        lmd = lmd+delta*(1-grand_delta)/norm_of_p_2 ;

                //    if (lmd > MAXFLOAT) lmd = MAXFLOAT ;
                        /* scg stops after 3 consecutive under m_dTolerance steps */
                if (epoch < lLength)
                {
                        k++;
                        epoch++;
                        for (i = 0; i < m_lNodesCount; i++)
                                m_pdNodes[i] = 0.0;
                        goto step2;
                }

        delete [] p;
        delete [] r;
        delete [] step;
    return;
}

void NeuralNet::testPattern(long lPattern){
                long pPattern = (m_plLayers[0]+m_plLayers[m_lLayersCount-1]) * lPattern;
                long pState;
                long pStatePrev;
                long pWeight;
                double Sum;

                long i,j,k;
                pState = 0;
                pWeight = 0;
                // Прямой процесс
                // Первый слой
                for (i = 0; i < m_plLayers[0]; i++)
                        m_pdNodes[pState++] = m_pdPatterns[pPattern++];
                // Остальные слои
                for (i = 1; i < m_lLayersCount; i++)
                {
                        for ( j = 0; j < m_plLayers[i]; j++)
                        {
                                Sum = m_pdWeights[pWeight++];
                                pStatePrev = m_plNodePtr[i-1];
                                for (k = 0; k < m_plLayers[i-1]; k++)
                                        Sum += m_pdNodes[pStatePrev++]*m_pdWeights[pWeight++];

                                if (0.0-Sum < DBL_MAX_10_EXP)
                                        m_pdNodes[pState++] = 1.0 / (1 + exp(0.0-Sum));
                                else
                                        m_pdNodes[pState++] = 0;
                        }
                }
                return;
}

void NeuralNet::createPatterns(long lPatterns)
{
        if (m_pdPatterns) delete[] m_pdPatterns;
        m_pdPatterns = new double[(m_plLayers[0]+m_plLayers[m_lLayersCount-1])*lPatterns];
        m_lPatternsCount = lPatterns;
}

double NeuralNet::getError()
{
	// Прямой процесс
	long    pStatePrev;     // Указатель на состояние нейрона предыдущего слоя
	long    pPattern = 0;      // Указатель на текущее значение паттерна обучения
	long    pState;
	long    pWeight;
	double  Sum;
	double  Error;
	double  ErrorSum = 0;

	long i, j, k, pattern;
	for (pattern = 0; pattern < m_lPatternsCount; pattern++)
	{
		pState = 0;
		pWeight = 0;
		// Прямой процесс
		// Первый слой
		for (i = 0; i < m_plLayers[0]; i++)
			m_pdNodes[pState++] = m_pdPatterns[pPattern++];
		// Остальные слои
		for (i = 1; i < m_lLayersCount; i++)
		{
			for (j = 0; j < m_plLayers[i]; j++)
			{
				Sum = m_pdWeights[pWeight++];
				pStatePrev = m_plNodePtr[i - 1];
				for (k = 0; k < m_plLayers[i - 1]; k++)
					Sum += m_pdNodes[pStatePrev++] * m_pdWeights[pWeight++];
				m_pdNodes[pState++] = 1.0 / (1 + exp(0.0 - Sum)); // Sigmoid function
			}
		}
		// Calculate Error
		pState = m_lNodesCount - 1;
		for (i = 0; i < m_plLayers[m_lLayersCount - 1]; i++)
		{
			Error = m_pdNodes[pState--] - m_pdPatterns[pPattern++];
			Error *= Error;
			ErrorSum += Error;
		}
	}// End of Pattern
	return ErrorSum;
}

double NeuralNet::getErrorWithoutUnit(long lNumUnit)
{
// Прямой процесс
        long    pStatePrev;     // Указатель на состояние нейрона предыдущего слоя
        long    pPattern = 0;      // Указатель на текущее значение паттерна обучения
        long    pState;
        long    pWeight;
        double  Sum;
        double  Error;
        double  ErrorSum = 0;

        long i,j,k,pattern;
        for (pattern = 0; pattern < m_lPatternsCount; pattern++)
        {
                pState = 0;
                pWeight = 0;
                // Прямой процесс
                // Первый слой
                for (i = 0; i < m_plLayers[0]; i++)
                        m_pdNodes[pState++] = m_pdPatterns[pPattern++];
                // Остальные слои
                for (i = 1; i < m_lLayersCount; i++)
                {
                        for ( j = 0; j < m_plLayers[i]; j++)
                        {
                                Sum = m_pdWeights[pWeight++];
                                pStatePrev = m_plNodePtr[i-1];
                                for (k = 0; k < m_plLayers[i-1]; k++)
                                        if (lNumUnit != pStatePrev)
                                                Sum += m_pdNodes[pStatePrev++]*m_pdWeights[pWeight++];
                                        else
                                        {
                                                pStatePrev++;
                                                pWeight++;
                                        }
                                m_pdNodes[pState++] = 1.0 / (1 + exp(0.0-Sum)); // Sigmoid function
                        }
                }
                // Calculate Error
                pState = m_lNodesCount-1;
                for (i = 0; i < m_plLayers[m_lLayersCount-1]; i++)
                {
                        Error = m_pdNodes[pState--] - m_pdPatterns[pPattern++];
                        Error *= Error;
                        ErrorSum += Error;
                }
        }// End of Pattern
        return ErrorSum;
}

double NeuralNet::GetRelevance(long lNode)
{
        double *pdRelevances = new double[m_lNodesCount];
        double dMaxRelevance;
        double dBaseError = getErrorWithoutUnit(-1);
		long i,j;
        for (i = 0; i < m_lNodesCount - m_plLayers[m_lLayersCount - 1]; i++)
                pdRelevances[i] = getErrorWithoutUnit(i) - dBaseError;

        for (j = 0; j < m_lLayersCount - 1; j++)
        {
                dMaxRelevance = 0;
                for (i = m_plNodePtr[j]; i < m_plNodePtr[j] + m_plLayers[j]; i++)
                        if (pdRelevances[i] > dMaxRelevance)
                                dMaxRelevance = pdRelevances[i];

                for (i = m_plNodePtr[j]; i < m_plNodePtr[j] + m_plLayers[j]; i++)
                        pdRelevances[i] /= dMaxRelevance;
        }
		double dResult = pdRelevances[lNode];
		delete [] pdRelevances;
		return dResult;
}

/*
CString NeuralNet::GetNMExpression(long lNode)
{
        long i,j,k,n,m,layer;
        long lNodesCount = 0;
        double * pdWeightPtr;
        double max,sum1,sum2,p,d,d_min = DBL_MAX,Eextr;
        long lmax;
		long m_N;
        long m_M;

        CString strExpression = "";
        CString strTemp = "";
        CString strNMExpression;

        if (lNode - m_plLayers[0] < 0 || lNode >= m_lNodesCount)
                return "Error Node Number";

        for (i=1;i<m_lLayersCount; i++)
        {
                lNodesCount += m_plLayers[i-1];
                if (lNode >= m_plNodePtr[i] && lNode < m_plNodePtr[i] + m_plLayers[i])
                {
                        layer = i;
                        k = m_plLayers[i-1];
                        break;
                }
        }
        layer--;
        double *u = new double[k+1];
        double *t = new double[k+1];

        // Указатель на текущие веса нейрона с номером lNode
        pdWeightPtr = m_pdWeights + m_plWeightPtr[layer] + (lNode - lNodesCount)*(m_plLayers[layer]+1);

        for (m = 1; m <= k; m++)
                for (n = 1; n <= m; n++)
                {
                        if (sign1(1+m-2*n)==sign1(pdWeightPtr[0]) || 1+m-2*n == 0)
                        {
								// 1.
                                for (i = 0; i<=k; i++)
                                        u[i] = 0.0;
                                // 2.
                                for (i = 0; i < m; i++)
                                {
                                        max = -0.1;
                                        lmax = 1;
                                        for (j = 1; j<=k; j++)
                                                if (fabs(pdWeightPtr[j]) > max && u[j] == 0)
                                                {
                                                        max = fabs(pdWeightPtr[j]);
                                                        lmax = j;
                                                }
                                        u[lmax] = sign1(pdWeightPtr[lmax]);
                                }
                                // 3.
                                u[0] = 1+m-2*n;
                                // 4.
                                sum1 = sum2 = 0;
                                for(i=0; i<=k;i++)
                                {
                                        sum1 += pdWeightPtr[i]*u[i];
                                        sum2 += u[i]*u[i];
                                }
                                p = sum1 / sum2;
                                // 5.
                                for(i=0; i<=k;i++)
                                        t[i] = u[i] * p;
                                // 6.
                                d = Eextr =0;
                                for(i=0; i<=k;i++)
                                {
                                        d += (t[i] - pdWeightPtr[i]) * (t[i] - pdWeightPtr[i]);
                                        Eextr += pdWeightPtr[i] * pdWeightPtr[i];
                                }
                                Eextr = 100 * d / Eextr;
                                // 7.
                                if (d < d_min)
                                {
                                        strExpression = "";
                                        d_min = d;
                                        for (i=1; i<=k; i++)
                                        {
                                                if(sign(t[i]<0))
                                                {
                                                        strTemp.Format("-U%ld  ",m_plNodePtr[layer]+i-1);
                                                        strExpression += strTemp;
                                                }
                                                else
                                                {
                                                        strTemp.Format("U%ld  ",m_plNodePtr[layer]+i-1);
                                                        strExpression += strTemp;
                                                }
                                        }
                                        strNMExpression.Format("%ld of ",n);
//                                        strNMExpression.Format("%ld of ",n);
                                        strNMExpression += "(  "+strExpression+")";
                                        m_M = m;
                                        m_N = n;
                                }
                        }
                }
		strTemp.Empty();
        strTemp.Format("U%ld = ",lNode);
		strNMExpression = strTemp + strNMExpression;
        delete [] u;
        delete [] t;
        return strNMExpression;
}
*/
