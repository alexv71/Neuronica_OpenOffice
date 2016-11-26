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

#include "stdafx.h"

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#include "som.h"
#include <float.h>
#include <math.h>

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif

// Comparison function for the sort
int compar(const void *first, const void *sec)
{
        if(*(double *)first < *(double *)sec) return -1;
        else return *(double *)first > *(double *)sec;
}

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
SOM::SOM()
{
        m_lXDim = 0;
        m_lYDim = 0;
        m_lPDim = 0;
}

SOM::~SOM()
{
}
//////////////////////////////////////////////////////////////////////
// Interface functions
//////////////////////////////////////////////////////////////////////
short SOM::CreateMap(long xDim, long yDim, long iDim, short Topology)
{
        if (xDim <= 0 || yDim <= 0 || iDim <= 0)
                return -1;
        m_lPatternsCount = 0;
        m_lXDim = xDim;
        m_lYDim = yDim;
        m_lPDim = iDim;
        m_vaMap.resize(m_lXDim * m_lYDim * m_lPDim);
        m_vaUmatrix.resize((2 * m_lXDim -1) * (2 * m_lYDim -1));
        m_vaKMeans.resize(m_lXDim * m_lYDim);
        m_vstrLabels.resize(m_lXDim * m_lYDim);
        m_vstrParamNames.resize(m_lPDim);
		m_vaMap = 0;
		m_vaUmatrix = 0;
	    m_sTopology = Topology;

        return 0;
}

short SOM::InitializeMap(short InitializationType, unsigned int RandomSeed)
{
        int x,y,z;

        if (m_lXDim <= 0 || m_lYDim <= 0 || m_lPDim <= 0)
                return -1;

        srand(RandomSeed);

        if (InitializationType == INIT_RND)
        {
				valarray<double> vaMaxVector(m_lPDim);
                valarray<double> vaMinVector(m_lPDim);

                for (z=0; z<m_lPDim;z++)
                {
						valarray<double> vaPatterns = m_vaPatterns[slice(z, m_lPatternsCount, m_lPDim)];
                        vaMaxVector[z] = vaPatterns.max();
                        vaMinVector[z] = vaPatterns.min();
                }

                for (x=0; x<m_lXDim;x++)
                        for (y=0; y<m_lYDim;y++)
							for (z=0; z<m_lPDim;z++)
								MAP(x,y,z) = vaMinVector[z] + (vaMaxVector[z] - vaMinVector[z]) * ((double)rand())/RAND_MAX;;

        }
        else
        {
                double xf, yf;
                long index = 0;
                // Find the middle point and two eigenvectors of the data
                FindEigenvectors();

                for (y=0; y<m_lYDim;y++)
                        for (x=0; x<m_lXDim;x++)
                        {
                                xf = 4.0 * (double)(index % m_lXDim) / (m_lXDim - 1.0) - 2.0;
                                yf = 4.0 * (double)(index / m_lXDim) / (m_lYDim - 1.0) - 2.0;
                                for (z=0; z<m_lPDim;z++)
                                        MAP(x,y,z) = m_vMean[z] + xf * m_vEigen1[z] + yf * m_vEigen2[z];

                                index++;
                        }
        }
        ClearLabels();
        return 0;
}

short SOM::InitializePatterns(long lPatternsCount)
{
        if (lPatternsCount <= 0 || m_lPDim <= 0)
                return -1;
        m_lPatternsCount = lPatternsCount;
        m_vaPatterns.resize(m_lPatternsCount * m_lPDim);
        m_pcMask.resize(m_lPatternsCount * m_lPDim);
        return 0;
}

short SOM::SetPatternValue(long lPatternNumber, long lPatternIndex, double lPatternValue, char cMask)
{
        if (lPatternNumber < 0 || lPatternNumber >= m_lPatternsCount)
                return -1;
        if (lPatternIndex < 0 || lPatternIndex >= m_lPDim)
                return -1;

        m_vaPatterns[lPatternNumber * m_lPDim + lPatternIndex] = lPatternValue;
        m_pcMask[lPatternNumber * m_lPDim + lPatternIndex] = cMask;
        return 0;
}


//////////////////////////////////////////////////////////////////////
// Calculating functions
//////////////////////////////////////////////////////////////////////
inline double SOM::GetHexaDist(long bx, long by, long tx, long ty)
{
        double temp, diff;

        diff = bx - tx;

        if (((by - ty) % 2) != 0) {
                if ((by % 2) == 0)
                        diff -= 0.5;
                else
                        diff += 0.5;
        }

        temp = diff * diff;
        diff = by - ty;
        temp += 0.75 * diff * diff;
        return(sqrt(temp));
}

inline double SOM::GetRectDist(long bx, long by, long tx, long ty)
{
        double temp, diff;

        diff = bx - tx;
        temp = diff * diff;
        diff = by - ty;
        temp += diff * diff;
        return(sqrt(temp));
}

void SOM::CalcUmatrix()
{
        int i,j,k,count,bx,by,bz;
        double dx,dy,dz1,dz2,dz,temp,max=0,min=0, bw;
        double medtable[6];
        double tmp;

        if (!m_lXDim || !m_lYDim || !m_lPDim)
                return;
// rectangular topology
        if (m_sTopology == TOPOL_RECT)
        {
                for (j=0;j<m_lYDim;j++)
                        for (i=0;i<m_lXDim;i++)
                        {
                                dx=0;dy=0;dz1=0;dz2=0;count=0;
                                bx=0;by=0;bz=0;
                                for (k=0;k<m_lPDim;k++)
                                {
                                        if (i<(m_lXDim-1))
                                        {
                                                temp = (MAP(i,j,k) - MAP(i+1,j,k));
                                                dx += temp*temp;
                                                bx=1;
                                        }
                                        if (j<(m_lYDim-1))
                                        {
                                                temp = (MAP(i,j,k) - MAP(i,j+1,k));
                                                dy += temp*temp;
                                                by=1;
                                        }
                                        if (j<(m_lYDim-1) && i<(m_lXDim-1))
                                        {
                                                temp = (MAP(i,j,k) - MAP(i+1,j+1,k));
                                                dz1 += temp*temp;
                                                temp = (MAP(i,j+1,k) - MAP(i+1,j,k));
                                                dz2 += temp*temp;
                                                bz=1;
                                        }
                                }
                                dz = (sqrt(dz1)/sqrt((double) 2.0)+sqrt(dz2)/sqrt((double) 2.0))/2;

                                if (bx)
                                        UMATRIX(2*i+1,2*j) = sqrt(dx);
                                if (by)
                                        UMATRIX(2*i,2*j+1) = sqrt(dy);
                                if (bz)
                                        UMATRIX(2*i+1,2*j+1) = dz;
                        }
        }
        else
// hexagonal topology
    {
                for (j=0;j<m_lYDim;j++)
                        for (i=0;i<m_lXDim;i++)
                        {
                                dx=0;dy=0;dz=0;count=0;
                                bx=0;by=0;bz=0;
                                temp=0;
                                if (i<(m_lXDim-1))
                                {
                                        for (k=0;k<m_lPDim;k++)
                                        {
                                                tmp = MAP(i,j,k);
                                                tmp = MAP(i+1,j,k);
                                                tmp = m_vaMap[5];
                                                temp = (MAP(i,j,k) - MAP(i+1,j,k));
                                                dx += temp*temp;
                                                bx=1;
                                        }
                                }
                                temp=0;
                                if (j<(m_lYDim-1))
                                {
                                        if (j%2)
                                        {
                                                for (k=0;k<m_lPDim;k++)
                                                {
                                                        temp = (MAP(i,j,k) - MAP(i,j+1,k));
                                                        dy += temp*temp;
                                                        by=1;
                                                }
                                        }
                                        else
                                        {
                                                if (i>0)
                                                {
                                                        for (k=0;k<m_lPDim;k++)
                                                        {
                                                                temp = (MAP(i,j,k) - MAP(i-1,j+1,k));
                                                                dy += temp*temp;
                                                                by=1;
                                                        }
                                                }
                                                else
                                                        temp=0;
                                        }
                                }
                                temp=0;
                                if (j<(m_lYDim-1))
                                {
                                        if (!(j%2))
                                        {
                                                for (k=0;k<m_lPDim;k++)
                                                {
                                                        temp = (MAP(i,j,k) - MAP(i,j+1,k));
                                                        dz += temp*temp;
                                                }
                                                bz=1;
                                        }
                                        else
                                        {
                                                if (i<(m_lXDim-1))
                                                {
                                                        for (k=0;k<m_lPDim;k++)
                                                        {
                                                                temp = (MAP(i,j,k) - MAP(i+1,j+1,k));
                                                                dz += temp*temp;
                                                        }
                                                        bz=1;
                                                }
                                        }
                                }
                                else
                                        temp=0;

                                if (bx)
                                        UMATRIX(2*i+1,2*j) = sqrt(dx);
                    if (by)
                                {
                                        if (j%2)
                                                UMATRIX(2*i,2*j+1) = sqrt(dy);
                                        else
                                                UMATRIX(2*i-1,2*j+1) = sqrt(dy);
                                }
                                if (bz)
                                {
                                        if (j%2)
                                                UMATRIX(2*i+1,2*j+1) = sqrt(dz);
                                        else
                                                UMATRIX(2*i,2*j+1) = sqrt(dz);
                                }
                        }
        }


// Set the values corresponding to the model vectors themselves
// to medians of the surrounding values
        if(m_sTopology == TOPOL_RECT)
        {
// medians of the 4-neighborhood
                for (j=0;j<m_lYDim * 2 - 1;j+=2)
                        for (i=0;i<m_lXDim * 2 - 1;i+=2)
                                if(i>0 && j>0 && i<m_lXDim * 2 - 1-1 && j<m_lYDim * 2 - 1-1)
                                {
// in the middle of the map
                                        medtable[0] = UMATRIX(i-1,j);
                                        medtable[1] = UMATRIX(i+1,j);
                                        medtable[2] = UMATRIX(i,j-1);
                                        medtable[3] = UMATRIX(i,j+1);
                                        qsort((void *)medtable, 4, sizeof(*medtable), compar);
// Actually mean of two median values
                                        UMATRIX(i,j)=(medtable[1]+medtable[2])/2.0;
                                }
                                else if(j==0 && i>0 && i<m_lXDim * 2 - 1-1)
                                {
// in the upper edge
                                        medtable[0]=UMATRIX(i-1,j);
                                        medtable[1]=UMATRIX(i+1,j);
                                        medtable[2]=UMATRIX(i,j+1);
                                        qsort((void *)medtable, 3, sizeof(*medtable), compar);
                                        UMATRIX(i,j)=medtable[1];
                                }
                                else if(j==m_lYDim * 2 - 1-1 && i>0 && i<m_lXDim * 2 - 1-1)
                                {
// in the lower edge
                                        medtable[0]=UMATRIX(i-1,j);
                                        medtable[1]=UMATRIX(i+1,j);
                                        medtable[2]=UMATRIX(i,j-1);
                                        qsort((void *)medtable, 3, sizeof(*medtable), compar);
                                        UMATRIX(i,j)=medtable[1];
                                }
                                else if(i==0 && j>0 && j<m_lYDim * 2 - 1-1)
                                {
// in the left edge
                                        medtable[0]=UMATRIX(i+1,j);
                                        medtable[1]=UMATRIX(i,j-1);
                                        medtable[2]=UMATRIX(i,j+1);
                                        qsort((void *)medtable, 3, sizeof(*medtable), compar);
                                        UMATRIX(i,j)=medtable[1];
                                }
                                else if(i==m_lXDim * 2 - 1-1 && j>0 && j<m_lYDim * 2 - 1-1)
                                {
// in the right edge
                                        medtable[0]=UMATRIX(i-1,j);
                                        medtable[1]=UMATRIX(i,j-1);
                                        medtable[2]=UMATRIX(i,j+1);
                                        qsort((void *)medtable, 3, sizeof(*medtable), compar);
                                        UMATRIX(i,j)=medtable[1];
                                }
                                else if(i==0 && j==0)
// the upper left-hand corner
                                        UMATRIX(i,j)=(UMATRIX(i+1,j)+UMATRIX(i,j+1))/2.0;
                                else if(i==m_lXDim * 2 - 1-1 && j==0)
                                {
// the upper right-hand corner
                                        UMATRIX(i,j)=(UMATRIX(i-1,j)+UMATRIX(i,j+1))/2.0;
                                }
                                else if(i==0 && j==m_lYDim * 2 - 1-1)
                                {
// the lower left-hand corner
                                        UMATRIX(i,j)=(UMATRIX(i+1,j)+UMATRIX(i,j-1))/2.0;
                                }
                                else if(i==m_lXDim * 2 - 1-1 && j==m_lYDim * 2 - 1-1)
                                {
// the lower right-hand corner
                                        UMATRIX(i,j)=(UMATRIX(i-1,j)+UMATRIX(i,j-1))/2.0;
                                }
        }
        else
// HEXA
                for (j=0;j<m_lYDim * 2 - 1;j+=2)
                        for (i=0;i<m_lXDim * 2 - 1;i+=2)
                                if(i>0 && j>0 && i<m_lXDim * 2 - 1-1 && j<m_lYDim * 2 - 1-1)
                                {
// in the middle of the map
                                        medtable[0]=UMATRIX(i-1,j);
                                        medtable[1]=UMATRIX(i+1,j);
                                        if(!(j%4))
                                        {
                                                medtable[2]=UMATRIX(i-1,j-1);
                                                medtable[3]=UMATRIX(i,j-1);
                                                medtable[4]=UMATRIX(i-1,j+1);
                                                medtable[5]=UMATRIX(i,j+1);
                                        }
                                        else
                                        {
                                                medtable[2]=UMATRIX(i,j-1);
                                                medtable[3]=UMATRIX(i+1,j-1);
                                                medtable[4]=UMATRIX(i,j+1);
                                                medtable[5]=UMATRIX(i+1,j+1);
                                        }
                                        qsort((void *)medtable, 6, sizeof(*medtable), compar);
// Actually mean of two median values
                                        UMATRIX(i,j)=(medtable[2]+medtable[3])/2.0;
                                }
                                else if(j==0 && i>0 && i<m_lXDim * 2 - 1-1)
                                {
// in the upper edge
                                        medtable[0]=UMATRIX(i-1,j);
                                        medtable[1]=UMATRIX(i+1,j);
                                        medtable[2]=UMATRIX(i,j+1);
                                        medtable[3]=UMATRIX(i-1,j+1);
                                        qsort((void *)medtable, 4, sizeof(*medtable), compar);
// Actually mean of two median values
                                        UMATRIX(i,j)=(medtable[1]+medtable[2])/2.0;
                                }
                                else if(j==m_lYDim * 2 - 1-1 && i>0 && i<m_lXDim * 2 - 1-1)
                                {
// in the lower edge
                                        medtable[0]=UMATRIX(i-1,j);
                                        medtable[1]=UMATRIX(i+1,j);
                                        if(!(j%4))
                                        {
                                                medtable[2]=UMATRIX(i-1,j-1);
                                                medtable[3]=UMATRIX(i,j-1);
                                        }
                                        else
                                        {
                                                medtable[2]=UMATRIX(i,j-1);
                                                medtable[3]=UMATRIX(i+1,j-1);
                                        }
                                        qsort((void *)medtable, 4, sizeof(*medtable), compar);
// Actually mean of two median values
                                        UMATRIX(i,j)=(medtable[1]+medtable[2])/2.0;
                                }
                                else if(i==0 && j>0 && j<m_lYDim * 2 - 1-1)
                                {
// in the left edge
                                        medtable[0]=UMATRIX(i+1,j);
                                        if(!(j%4))
                                        {
                                                medtable[1]=UMATRIX(i,j-1);
                                                medtable[2]=UMATRIX(i,j+1);
                                                qsort((void *)medtable, 3, sizeof(*medtable), compar);
                                                UMATRIX(i,j)=medtable[1];
                                        }
                                        else
                                        {
                                                medtable[1]=UMATRIX(i,j-1);
                                                medtable[2]=UMATRIX(i+1,j-1);
                                                medtable[3]=UMATRIX(i,j+1);
                                                medtable[4]=UMATRIX(i+1,j+1);
                                                qsort((void *)medtable, 5, sizeof(*medtable), compar);
                                                UMATRIX(i,j)=medtable[2];
                                        }
                                }
                                else if(i==m_lXDim * 2 - 1-1 && j>0 && j<m_lYDim * 2 - 1-1)
                                {
// in the right edge
                                        medtable[0]=UMATRIX(i-1,j);
                                        if(j%4)
                                        {
                                                medtable[1]=UMATRIX(i,j-1);
                                                medtable[2]=UMATRIX(i,j+1);
                                                qsort((void *)medtable, 3, sizeof(*medtable), compar);
                                                UMATRIX(i,j)=medtable[1];
                                        }
                                        else
                                        {
                                                medtable[1]=UMATRIX(i,j-1);
                                                medtable[2]=UMATRIX(i-1,j-1);
                                                medtable[3]=UMATRIX(i,j+1);
                                                medtable[4]=UMATRIX(i-1,j+1);
                                                qsort((void *)medtable, 5, sizeof(*medtable), compar);
                                                UMATRIX(i,j)=medtable[2];
                                        }
                                }
                                else if(i==0 && j==0)
// the upper left-hand corner
                                        UMATRIX(i,j)=(UMATRIX(i+1,j)+UMATRIX(i,j+1))/2.0;
                                else if(i==m_lXDim * 2 - 1-1 && j==0)
                                {
// the upper right-hand corner
                                        medtable[0]=UMATRIX(i-1,j);
                                        medtable[1]=UMATRIX(i-1,j+1);
                                        medtable[2]=UMATRIX(i,j+1);
                                        qsort((void *)medtable, 3, sizeof(*medtable), compar);
                                        UMATRIX(i,j)=medtable[1];
                                }
                                else if(i==0 && j==m_lYDim * 2 - 1-1)
                                {
// the lower left-hand corner
                                        if(!(j%4))
                                                UMATRIX(i,j)=(UMATRIX(i+1,j)+UMATRIX(i,j-1))/2.0;
                                        else
                                        {
                                                medtable[0]=UMATRIX(i+1,j);
                                                medtable[1]=UMATRIX(i,j-1);
                                                medtable[2]=UMATRIX(i+1,j-1);
                                                qsort((void *)medtable, 3, sizeof(*medtable), compar);
                                                UMATRIX(i,j)=medtable[1];
                                        }
                                }
                                else if(i==m_lXDim * 2 - 1-1 && j==m_lYDim * 2 - 1-1)
                                {
// the lower right-hand corner
                                        if(j%4)
                                                UMATRIX(i,j)=(UMATRIX(i-1,j)+UMATRIX(i,j-1))/2.0;
                                        else
                                        {
                                                medtable[0]=UMATRIX(i-1,j);
                                                medtable[1]=UMATRIX(i,j-1);
                                                medtable[2]=UMATRIX(i-1,j-1);
                                                qsort((void *)medtable, 3, sizeof(*medtable), compar);
                                                UMATRIX(i,j)=medtable[1];
                                        }
                                }

// scale values to (0,1)

        bw = m_vaUmatrix.max() - m_vaUmatrix.min();
        min = m_vaUmatrix.min();
//      m_vaUmatrix = (m_vaUmatrix - min) / bw;
        for (i=0;i<m_lXDim * 2 - 1;i++)
                for (j=0;j<m_lYDim * 2 - 1;j++)
                        UMATRIX(i,j) = (UMATRIX(i,j) - min) / bw;

        return;
}

inline void SOM::Normalize(valarray<double> &v, long lStartIndex)
{
        double sum = sqrt(inner_product(&v[lStartIndex],&v[lStartIndex+m_lPDim],&v[lStartIndex],0.0));

        for (long i=lStartIndex; i<lStartIndex+m_lPDim; i++)
                v[i] /= sum;

        return;
}

void SOM::GramSchmidt(valarray <double> &v, long n, long e)
{
        int i, j, p, t;
        double sum;
        valarray <double> w(0.0, n*e);

        for (i=0; i<e; i++)
        {
                for (t=0; t<n; t++)
                {
                        sum=v[i*n+t];
                        for (j=0; j<i; j++)
                                for (p=0; p<n; p++)
                                        sum-=w[j*n+t]*w[j*n+p]*v[i*n+p];

                        w[i*n+t]=sum;
                }
                Normalize(w,i*n);
        }
        v = w;
        return;
}

void SOM::AdaptBubble(long bx, long by, double radius, double alpha, long PatternIndex)
{
        PDist Dist;

        if (m_sTopology == TOPOL_RECT)
                Dist = &SOM::GetRectDist;
        else
                Dist = &SOM::GetHexaDist;

        for(long x=0; x<m_lXDim; x++)
                for(long y=0; y<m_lYDim; y++)
                {
                        if ((this->*Dist)(bx, by, x, y) <= radius)
                                for (long z = 0; z < m_lPDim; z++)
//                                        if (m_pcMask[PatternIndex+z])
                                                MAP(x,y,z) += alpha * (m_vaPatterns[PatternIndex+z] - MAP(x,y,z));
                }
}

void SOM::AdaptGaussian(long bx, long by, double radius, double alpha, long PatternIndex)
{
        double dd;
        double alp;
        PDist Dist;

        if (m_sTopology == TOPOL_RECT)
                Dist = &SOM::GetRectDist;
        else
                Dist = &SOM::GetHexaDist;

        for(long x=0; x<m_lXDim; x++)
                for(long y=0; y<m_lYDim; y++)
                {
                        dd = (this->*Dist)(bx, by, x, y);

                        alp = alpha * (double) exp((double) (-dd * dd / (2.0 * radius * radius)));

                        for (long z = 0; z < m_lPDim; z++)
//                                if (m_pcMask[PatternIndex+z])
                                        MAP(x,y,z) += alp * (m_vaPatterns[PatternIndex+z] - MAP(x,y,z));
                }
}

void SOM::FindEigenvectors()
{
        int n=m_lPDim;

        valarray<double> r(0.0, m_lPDim * m_lPDim);
        valarray<double> m(0.0, m_lPDim);
        valarray<double> u(0.0, m_lPDim * 2);
        valarray<double> v(0.0, m_lPDim * 2);
        valarray<double> k2(0.0, m_lPDim);
        double mu[2];
        double sum;
        int i,j;
        long k;

        for (k=0; k<m_lPatternsCount; k++)
                for (i=0; i<n; i++)
// masked components have the value 0 so they don't affect the sum
                if (m_pcMask[k*n+i])
                {
                        m[i]+=m_vaPatterns[k*n+i];
                        k2[i]++;
                }

//if (k<3) goto everror;

        m /= k2;

        for (k=0; k<m_lPatternsCount; k++)
        {
                for (i=0; i<n; i++)
                {
// the components that are masked off are ignored
                        if (!m_pcMask[k*n+i])
                                continue;
                        for (j=i; j<n; j++)
                                if (!m_pcMask[k*n+j])
                                        continue;
                                else
                                        r[i*n+j]+=(m_vaPatterns[k*n+i]-m[i])*(m_vaPatterns[k*n+j]-m[j]);
                }
        }




        for (i=0; i<n; i++)
                for (j=i; j<n; j++)
                        r[j*n+i]=r[i*n+j]/=m_lPatternsCount;


        for (i=0; i<2; i++)
        {
                for (j=0; j<n; j++)
                        u[i*n+j]=((double)rand())/RAND_MAX*2.0-1.0;

                Normalize(u,i*n);
                mu[i]=1.0;
        }



        for (k=0; k<10; k++)
        {
                for (i=0; i<2; i++)
                        for (j=0; j<n; j++)
                                v[i*n+j] = mu[i] * inner_product(&r[j*n], &r[j*n+n], &u[i*n], 0.0) + u[i*n+j];

                GramSchmidt(v, n, 2);
                sum=0.0;

                for (i=0; i<2; i++)
                {
                        for (j=0; j<n; j++)
                                if (inner_product(&r[j*n], &r[j*n+n], &v[i*n], 0.0)!=0)
                                        sum += fabs(v[i*n+j] / inner_product(&r[j*n], &r[j*n+n], &v[i*n], 0.0));

                        mu[i]=sum/n;
                }
                u = v;
        }

//  if (mu[0]==0.0 || mu[1]==0.0) goto everror;

        m_vMean.resize(m_lPDim);
        m_vEigen1.resize(m_lPDim);
        m_vEigen2.resize(m_lPDim);

        m_vMean = m;
    for (j=0; j<n; j++)
                m_vEigen1[j] = u[j] / sqrt(mu[0]);

    for (j=0; j<n; j++)
                m_vEigen2[j] = u[j+m_lPDim] / sqrt(mu[1]);

//      m_vEigen1 /= sqrt(mu[0]);
//      m_vEigen2 /= sqrt(mu[1]);
}

void SOM::Train(long Length, double Alpha, double Radius, short AlphaType, short NeighbourhoodType, long TrainEventInterval)
{
        PAdapt AdaptFunc;
        PAlpha AlphaFunc;
        long XWin, YWin;
//double weight;
        double trad, talp;
        long le;
        double Distance;
        long p;
        long ep = Length * m_lPatternsCount;

        if (NeighbourhoodType == ADAPT_BUBBLE)
                AdaptFunc = &SOM::AdaptBubble;
        else
                AdaptFunc = &SOM::AdaptGaussian;

        if (AlphaType == ALPHA_LINEAR)
                AlphaFunc = &SOM::AlphaLinear;
        else
                AlphaFunc = &SOM::AlphaInverseT;

		for (le = 0, p = 0; le < ep; le++, p++)
        {
			if (p == m_lPatternsCount) {
				p = 0;
				if (TrainEventInterval>0 && (le/m_lPatternsCount)%TrainEventInterval==0) 
					FireTrainEvent();
			}
// Radius decreases linearly to one
                trad = 1.0 + (Radius - 1.0) * (double) (ep - le) / (double) ep;
                talp = (this->*AlphaFunc)(le, ep, Alpha);

                XWin = YWin = 0;
                FindWinner(p,XWin,YWin,Distance);
//                if(FindWinner(p,XWin,YWin,Distance)!=0)
//                        continue;
                (this->*AdaptFunc)(XWin, YWin, trad, talp, p*m_lPDim);
        }
        return;
}

double SOM::AlphaLinear(long Iter, long Epoches, double Alpha)
{
        return (Alpha * (double) (Epoches - Iter) / (double) Epoches);
}

double SOM::AlphaInverseT(long Iter, long Epoches, double Alpha)
{
        double c = (double)Epoches / INV_ALPHA_CONSTANT;
        return (Alpha * c / (c + Iter));
}

double SOM::FindQError()
{
        double Distance;
        long XWin, YWin;
        double QError = 0.0;

        for(long p=0; p<m_lPatternsCount; p++)
        {
                if(FindWinner(p,XWin,YWin,Distance)!=0)
                        continue;
                QError += sqrt(Distance);
        }

        if (m_lPatternsCount && m_lXDim && m_lYDim && m_lPDim)
                return(QError/(double)m_lPatternsCount);
        else
                return -1;
}


void SOM::ClearLabels()
{
        for (long i=0; i<m_lXDim*m_lYDim; i++)
                m_vstrLabels[i] = "";
}

short SOM::FindWinner(long PatternNumber, long &XWin, long &YWin, double &Distance)
{
        long Masked;//, NeedBreak;
        double Diff, Difference;

        if (PatternNumber >= m_lPatternsCount || PatternNumber < 0)
                return -1;

        Distance = DBL_MAX;
//        NeedBreak = 0;
// Compute the distance between codebook and input entry
        for(long x=0; x<m_lXDim; x++)
                for(long y=0; y<m_lYDim; y++)
                {
                        Masked = 0;
                        Difference = 0.0;
                        for (long z = 0; z < m_lPDim; z++)
                        {
//                                if (m_pcMask[PatternNumber*m_lPDim+z])
//                                {
                                        Diff = MAP(x,y,z) - m_vaPatterns[PatternNumber*m_lPDim+z];
                                        Difference += Diff * Diff;
                                      if (Difference > Distance)
                                              break;
//                                }
//                                else
//                                        Masked++;
                        }
// ignore empty samples
//                        if (Masked == m_lPDim)
//                                NeedBreak = 1;

// If distance is smaller than previous distances
                        if (Difference < Distance)
                        {
                                XWin = x;
                                YWin = y;
                                Distance = Difference;
                        }
                }

//        if (NeedBreak)
//                return -1;

        return 0;
}

double SOM::GetDistance(long PatternNumber, long UnitNumber)
{
        if (PatternNumber < 0 || PatternNumber >= m_lPatternsCount)
                return -1;

        if (UnitNumber < 0 || UnitNumber >= m_lXDim * m_lYDim)
                return -1;

        double Diff, Difference = 0;
        long x = UnitNumber % m_lXDim;
        long y = UnitNumber / m_lXDim;

        for (long z = 0; z < m_lPDim; z++)
        {
                if (m_pcMask[PatternNumber*m_lPDim+z])
                {
                        Diff = MAP(x,y,z) - m_vaPatterns[PatternNumber*m_lPDim+z];
                        Difference += Diff * Diff;
                }
        }
        return sqrt(Difference);
}

string SOM::GetLabel(long UnitNumber)
{
        if (UnitNumber >= m_lXDim*m_lYDim || UnitNumber < 0)
                return "";

        return m_vstrLabels[UnitNumber];
}

short SOM::SetLabel(long UnitNumber, string Value)
{
        if (UnitNumber >= m_lXDim*m_lYDim || UnitNumber < 0)
                return -1;

        m_vstrLabels[UnitNumber] = Value;
        return 0;
}

string SOM::GetParamName(long ParamNumber)
{
        if (ParamNumber >= m_lPDim || ParamNumber < 0)
                return "";

        return m_vstrParamNames[ParamNumber];
}

short SOM::SetParamName(long ParamNumber, string Value)
{
        if (ParamNumber >= m_lPDim || ParamNumber < 0)
                return -1;

        m_vstrParamNames[ParamNumber] = Value;
        return 0;
}

void SOM::CalcKMeans(long lIterationsCount, long lClustersCount)
{
        long xI, xJ;
        long xNum = lIterationsCount % 2;

        for(xI=0; xI<lClustersCount; xI++)
        {
                for(xJ=0; xJ<m_lPDim; xJ++)
                {
                }
        }
}

void SOM::FireTrainEvent(void)
{
}
