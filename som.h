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

#include <valarray>
#include <string>
#include <vector>
#include <numeric>

using namespace std;

//////////////////////////////////////////////////////////////////////
// Macroses and constants
//////////////////////////////////////////////////////////////////////

#define MAP(x,y,z) m_vaMap[((y)*m_lXDim + (x)) * m_lPDim + (z)]
#define UMATRIX(x,y) m_vaUmatrix[(y)*(m_lXDim*2-1) + (x)]

const double INV_ALPHA_CONSTANT = 100.0;

const long TOPOL_HEXA = 0;
const long TOPOL_RECT = 1;

const long INIT_LIN = 0;
const long INIT_RND = 1;

const long ADAPT_BUBBLE = 0;
const long ADAPT_GAUSSIAN = 1;

const long ALPHA_LINEAR = 0;
const long ALPHA_INVERSE_T = 1;

class SOM
{
public:
        SOM();
        virtual ~SOM();
        short SetParamName(long ParamNumber, string Value);
        string GetParamName(long ParamNumber);
        void CalcKMeans(long lIterationsCount, long lClustersCount);
        short SetLabel(long UnitNumber, string Value);
        double GetDistance(long PatternNumber, long UnitNumber);
        string GetLabel(long UnitNumber);
        short FindWinner(long PatternNumber, long &XWin, long &YWin, double &Distance);
        void ClearLabels();
        double FindQError();
        void Train(long Length, double Alpha, double Radius, short AlphaType, short NeighbourhoodType, long TrainEventInterval);
        short SetPatternValue(long lPatternNumber, long lPatternIndex, double lPatternValue, char cMask);
        short InitializePatterns(long lPatternsCount);
        void CalcUmatrix();
        short CreateMap(long xDim, long yDim, long iDim, short Topology);
        short InitializeMap(short InitializationType, unsigned int RandomSeed);

protected:
        vector <string> m_vstrLabels;
        vector <string> m_vstrParamNames;
        valarray<double> m_vaPatterns;
        valarray<double> m_vaUmatrix;
        valarray<double> m_vaKMeans;
        valarray<double> m_vaMap;
        vector<char> m_pcMask;
        long m_lXDim;
        long m_lYDim;
        long m_lPDim;
        long m_lPatternsCount;
        short m_sTopology;
		virtual void FireTrainEvent(void);

private:
        inline double AlphaInverseT(long Iter, long Epoches, double Alpha);
        inline double AlphaLinear(long Iter, long Epoches, double Alpha);
        inline void AdaptBubble(long bx, long by, double radius, double alpha, long PatternIndex);
        inline void AdaptGaussian(long bx, long by, double radius, double alpha, long PatternIndex);
        inline double GetRectDist(long bx, long by, long tx, long ty);
        inline double GetHexaDist(long bx, long by, long tx, long ty);
        valarray<double> m_vMean;
        valarray<double> m_vEigen1;
        valarray<double> m_vEigen2;
        inline void FindEigenvectors();
        inline void Normalize(valarray<double> &v, long lStartIndex);
        inline void GramSchmidt(valarray <double> &v, long n, long e);
};

typedef double (SOM::*PDist)(long bx, long by, long tx, long ty);
typedef void (SOM::*PAdapt)(long bx, long by, double radius, double alpha, long PatternIndex);
typedef double (SOM::*PAlpha)(long Iter, long Epoches, double Alpha);
