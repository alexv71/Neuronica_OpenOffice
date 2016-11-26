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

#include <cppuhelper/implbase3.hxx> // "3" implementing three interfaces
#include <cppuhelper/factory.hxx>
#include <cppuhelper/implementationentry.hxx>

#include <com/sun/star/lang/XServiceInfo.hpp>
#include <com/sun/star/lang/XInitialization.hpp>
#include <com/sun/star/lang/IllegalArgumentException.hpp>
#include <neuronica/XBackProp.hpp>
#include "neuralnet.h"


using namespace ::rtl; // for OUString
using namespace ::com::sun::star; // for odk interfaces
using namespace ::com::sun::star::uno; // for basic types


namespace my_sc_impl
{

static Sequence< OUString > getSupportedServiceNames_BackPropImpl()
{
    Sequence<OUString> names(1);
    names[0] = OUString(RTL_CONSTASCII_USTRINGPARAM("neuronica.BackProp"));
    return names;
}

static OUString getImplementationName_BackPropImpl()
{
    return OUString( RTL_CONSTASCII_USTRINGPARAM(
                         "neuronica.my_sc_implementation.BackProp") );
}
    
class BackPropImpl : public ::cppu::WeakImplHelper3<
      ::neuronica::XBackProp, lang::XServiceInfo, lang::XInitialization >,
	  public NeuralNet
{
    // it's good practise to store the context for further use when you use
    // other UNO API's in your implementation
    Reference< XComponentContext > m_xContext;
public:
    inline BackPropImpl(Reference< XComponentContext > const & xContext) throw ()
        : m_xContext(xContext)
        {}

    virtual ~BackPropImpl() {}

    // focus on three given interfaces,
    // no need to implement XInterface, XTypeProvider, XWeak
    
    // XInitialization will be called upon
    // createInstanceWithArguments[AndContext]()

    virtual void SAL_CALL initialize( Sequence< Any > const & args )
        throw (Exception);

	// XBackProp
	virtual sal_Int16 SAL_CALL createNetwork3(sal_Int32 il, sal_Int32 hl, sal_Int32 ol)
		throw (RuntimeException);
	virtual void SAL_CALL initializeNetwork(sal_Int32 randomSeed)
		throw (RuntimeException);
	virtual sal_Int32 SAL_CALL getLayersCount()
		throw (RuntimeException);
	virtual sal_Int32 SAL_CALL getWeightsCount()
		throw (RuntimeException);

	virtual sal_Int32 SAL_CALL getLayer(sal_Int32 nLayer)
		throw (RuntimeException);
	virtual double SAL_CALL getWeight(sal_Int32 nWeight)
		throw (RuntimeException);
	virtual void SAL_CALL setWeight(sal_Int32 nWeight, double dValue)
		throw (RuntimeException);

	virtual void SAL_CALL createPatterns(sal_Int32 lPatterns)
		throw (RuntimeException);
	virtual double SAL_CALL getPatternValue(sal_Int32 lPattern, sal_Int32 lUnit)
		throw (RuntimeException);
	virtual void SAL_CALL setPatternValue(sal_Int32 lPattern, sal_Int32 lUnit, double dValue)
		throw (RuntimeException);

	virtual void SAL_CALL trainBPROP(sal_Int32 lLength, double dMaxError, double dLearningRate, double dMomentum)
		throw (RuntimeException);
	virtual void SAL_CALL trainRPROP(sal_Int32 lLength, double dMaxError, double dIncreaseFactor, double dDecreaseFactor, double dDeltaMin, double dDeltaMax, double dDeltaInit)
		throw (RuntimeException);
	virtual void SAL_CALL trainSCG(sal_Int32 lLength, double dMaxError, double dSigma, double dLambda)
		throw (RuntimeException);
	virtual void SAL_CALL test(sal_Int32 lPattern)
		throw (RuntimeException);
	virtual double SAL_CALL getActivation(sal_Int32 nIndex)
		throw (RuntimeException);
	virtual double SAL_CALL getError()
		throw (RuntimeException);

	// XServiceInfo
    virtual OUString SAL_CALL getImplementationName()
        throw (RuntimeException);
    virtual sal_Bool SAL_CALL supportsService( OUString const & serviceName )
        throw (RuntimeException);
    virtual Sequence< OUString > SAL_CALL getSupportedServiceNames()
        throw (RuntimeException);
};

// XInitialization implemention

void BackPropImpl::initialize( Sequence< Any > const & args )
    throw (Exception)
{
}


// XBackProp implementation 
sal_Int16 BackPropImpl::createNetwork3(sal_Int32 il, sal_Int32 hl, sal_Int32 ol)
throw (RuntimeException)
{
	long i_array[3];
	sal_Int16 result = 0;
	i_array[0] = il; i_array[1] = hl; i_array[2] = ol;
	result = NeuralNet::createNetwork(3, i_array);
	//	delete i_array;

	return result;
}

void BackPropImpl::initializeNetwork(sal_Int32 randomSeed)
throw (RuntimeException)
{
	NeuralNet::initializeNetwork(randomSeed);
	return;
}

sal_Int32 BackPropImpl::getLayersCount()
throw (RuntimeException)
{
	return (sal_Int32)m_lLayersCount;
}

sal_Int32 BackPropImpl::getWeightsCount()
throw (RuntimeException)
{
	return (sal_Int32)m_lWeightsCount;
}

sal_Int32 BackPropImpl::getLayer(sal_Int32 nLayer)
{
	return (sal_Int32)m_plLayers[nLayer];
}

double BackPropImpl::getWeight(sal_Int32 nWeight)
{
	return m_pdWeights[nWeight];
}

void BackPropImpl::setWeight(sal_Int32 nWeight, double dValue)
{
	m_pdWeights[nWeight] = dValue;
}

void BackPropImpl::createPatterns(sal_Int32 lPatterns)
{
	NeuralNet::createPatterns(lPatterns);
}

double BackPropImpl::getPatternValue(sal_Int32 lPattern, sal_Int32 lUnit)
{
	return m_pdPatterns[(m_plLayers[0] + m_plLayers[m_lLayersCount - 1]) * lPattern + lUnit];
}

void BackPropImpl::setPatternValue(sal_Int32 lPattern, sal_Int32 lUnit, double dValue)
{
	NeuralNet::setPattern(lPattern, lUnit, dValue);
}

void BackPropImpl::trainBPROP(sal_Int32 lLength, double dMaxError, double dLearningRate, double dMomentum)
{
	NeuralNet::trainBPROP(lLength, dMaxError, dLearningRate, dMomentum);
}

void BackPropImpl::trainRPROP(sal_Int32 lLength, double dMaxError, double dIncreaseFactor, double dDecreaseFactor, double dDeltaMin, double dDeltaMax, double dDeltaInit)
{
	NeuralNet::trainRPROP(lLength, dMaxError, dIncreaseFactor, dDecreaseFactor, dDeltaMin, dDeltaMax, dDeltaInit);
}

void BackPropImpl::trainSCG(sal_Int32 lLength, double dMaxError, double dSigma, double dLambda)
{
	NeuralNet::trainSCG(lLength, dMaxError, dSigma, dLambda);
}

void BackPropImpl::test(sal_Int32 lPattern)
{
	NeuralNet::testPattern(lPattern);
}

double BackPropImpl::getActivation(sal_Int32 nIndex)
{
	if (nIndex >= 0 && nIndex < m_lNodesCount)
		return m_pdNodes[nIndex];
	else
		return -1;
}

double BackPropImpl::getError()
{
	return NeuralNet::getError();
}


// XServiceInfo implementation
OUString BackPropImpl::getImplementationName()
    throw (RuntimeException)
{
    // unique implementation name
    return OUString( RTL_CONSTASCII_USTRINGPARAM(
                         "neuronica.my_sc_implementation.BackProp") );
}

sal_Bool BackPropImpl::supportsService( OUString const & serviceName )
    throw (RuntimeException)
{
    // this object only supports one service, so the test is simple
    return serviceName.equalsAsciiL( RTL_CONSTASCII_STRINGPARAM("neuronica.BackProp") );
}

Sequence< OUString > BackPropImpl::getSupportedServiceNames()
    throw (RuntimeException)
{
    return getSupportedServiceNames_BackPropImpl();
}

Reference< XInterface > SAL_CALL create_BackPropImpl(
    Reference< XComponentContext > const & xContext )
    SAL_THROW( () )
{
    return static_cast< ::cppu::OWeakObject * >( new BackPropImpl( xContext ) );
}

}

/* shared lib exports implemented without helpers in service_impl1.cxx */
namespace my_sc_impl
{
static struct ::cppu::ImplementationEntry s_component_entries [] =
{
    {
        create_BackPropImpl, getImplementationName_BackPropImpl,
        getSupportedServiceNames_BackPropImpl,
        ::cppu::createSingleComponentFactory,
        0, 0
    },
    { 0, 0, 0, 0, 0, 0 }
};
}

extern "C"
{
SAL_DLLPUBLIC_EXPORT void SAL_CALL component_getImplementationEnvironment(
    sal_Char const ** ppEnvTypeName, uno_Environment ** )
{
    *ppEnvTypeName = CPPU_CURRENT_LANGUAGE_BINDING_NAME;
}

// This method not longer necessary since OOo 3.4 where the component registration was
// was changed to passive component registration. For more details see
// http://wiki.services.openoffice.org/wiki/Passive_Component_Registration
//
// sal_Bool SAL_CALL component_writeInfo(
//     lang::XMultiServiceFactory * xMgr, registry::XRegistryKey * xRegistry )
// {
//     return ::cppu::component_writeInfoHelper(
//         xMgr, xRegistry, ::my_sc_impl::s_component_entries );
// }


SAL_DLLPUBLIC_EXPORT void * SAL_CALL component_getFactory(
    sal_Char const * implName, lang::XMultiServiceFactory * xMgr,
    registry::XRegistryKey * xRegistry )
{
    return ::cppu::component_getFactoryHelper(
        implName, xMgr, xRegistry, ::my_sc_impl::s_component_entries );
}

}


