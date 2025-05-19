//                  MFEM Example 18 - Serial/Parallel Shared Code for MHD equations
//                      (Implementation of Time-dependent DG Operator)
//
// This code provide example problems for the MHD equations and implements
// the time-dependent DG operator given by the equation:
//
//            (u_t, v)_T - (F(u), ∇ v)_T + <F̂(u, n), [[v]]>_F = 0.
//
// This operator is designed for explicit time stepping methods. Specifically,
// the function DG_MHD_GPsource::Mult implements the following
// transformation:
//
//                             u ↦ M⁻¹(-DF(u) + NF(u))
//
// where M is the mass matrix, DF is the weak divergence of flux, and NF is the
// interface flux. The inverse of the mass matrix is computed element-wise by
// leveraging the block-diagonal structure of the DG mass matrix. Additionally,
// the flux-related terms are computed using the HyperbolicFormIntegrator.
//
// The maximum characteristic speed is determined for each time step. For more
// details, refer to the documentation of DG_MHD_GPsource::Mult.
//

#include <functional>
#include "mfem.hpp"

namespace mfem
{

/// @brief Godunov-Powell source term
class GPsource
{
public:
    const int num_equations;
    const int dim;

    /// @brief Constructor for the Godunov-Powell source term
    GPsource(const int dim)
    : num_equations(2*dim+2), dim(dim) { }

    /// @brief Destructor for the Godunov-Powell source term
    ~GPsource() {}

    /**
    * @brief Compute source -S(U)*divB in cell interiors.
    *
    * Used in GPsourceFormIntegrator::AssembleElementVector() for evaluation
    * of (-S(U)*divB, v) and in the default implementation of ComputeSourceDotN()
    * for evaluation of -S(U)*(Bhat-B).
    * @param[in] state state at the current integration point (num_equations)
    * @param[in] divB divergence of the magnetic field at the current integration point (scalar)
    * @param[in] Tr element transformation
    * @param[out] source source from the given element at the current integration point (num_equations)
    * @return real_t maximum characteristic speed |dF(u,x)/du|
    *
    * @note One can put assertion in here to detect non-physical solution
    */
    void ComputeSource(const Vector &state, const real_t &divB, ElementTransformation &Tr, Vector &source) const
    {
        source.SetSize(num_equations);
        source(0) = 0.0;
        for (auto i=0;i<dim;i++) source(i+1) = -state(i+1+dim) * divB;
        for (auto i=0;i<dim;i++) source(i+1+dim) = -state(i+1) / state(0) * divB;
        source(1+2*dim) = 0.0;
        for (auto i=0;i<dim;i++) source(1+2*dim) -= (state(i+1) / state(0)) * state(i+1+dim);
        source(1+2*dim) *= divB;
        return;
    }
};

class NumericalGPsource
{
protected:
    const GPsource &gpSource;
public:
    NumericalGPsource(const GPsource &gpSource)
      : gpSource(gpSource) { }
    
    virtual ~NumericalGPsource() = default;

    /**
    * @brief Evaluates normal numerical source for the given states and normal.
    * Must be implemented in a derived class.
    *
    * Used in HyperbolicFormIntegrator::AssembleFaceVector() for evaluation of
    * <S(U^-)*(\hat{B} - B^-)*n^-, v^-> term at the face.
    * @param[in] state1 state value at a point from the first element
    * (num_equations)
    * @param[in] state2 state value at a point from the second element
    * (num_equations)
    * @param[in] nor scaled normal vector, see mfem::CalcOrtho() (dim)
    * @param[in] Tr face transformation
    * @param[out] source1, source2 numerical source on both sides (not equal) (num_equations)
    */
    virtual void Eval(const Vector &state1, const Vector &state2, const Vector &nor, FaceElementTransformations &Tr, Vector &source1, Vector &source2) const = 0;

    /**
    * @brief Evaluates Jacobian of the normal numerical source for the given
    * states and normal. OPTIONALLY overloaded in a derived class.
    *
    * Used in HyperbolicFormIntegrator::AssembleFaceGrad() for Jacobian
    * of the term <S(U^-)*(\hat{B} - B^-)*n^-, v^-> at the face.
    * @param[in] side indicates gradient w.r.t. the first (side = 1)
    * or second (side = 2) state
    * @param[in] state1 state value of the beginning of the interval
    * (num_equations)
    * @param[in] state2 state value of the end of the interval
    * (num_equations)
    * @param[in] nor scaled normal vector, see mfem::CalcOrtho() (dim)
    * @param[in] Tr face transformation
    * @param[out] grad Jacobian of normal numerical flux (num_equations, dim)
    */
    virtual void Grad(int side, const Vector &state1, const Vector &state2,
        const Vector &nor, FaceElementTransformations &Tr,
        DenseMatrix &grad) const
    { MFEM_ABORT("Not implemented."); }

    /// @brief Get source function
    /// @return constant reference to the source function.
    const GPsource &GetSourceFunction() const { return gpSource; }

};

class RusanovGPsource : public NumericalGPsource
{
public:
    RusanovGPsource(const GPsource &gpSource)
      : NumericalGPsource(gpSource) { }

    // at a face quadrature node with normal vector "nor" (not a unit vector) pointing from el1 to el2, 
    // compute source1 that is to be tested by local test function from el1, directly (coefficient is +1)
    // and source2 that is to be tested by local test function from el2, directly (coefficient is also +1)
    void Eval(const Vector &state1, const Vector &state2, const Vector &nor, FaceElementTransformations &Tr, Vector &source1, Vector &source2) const override
    {
        const auto dim = gpSource.dim;
        // nor points from el1 to el2, when they are obtained through "GetFaceElementTransformations"
        // inner product with "nor" (nor is not a unit vector, just use a form consistent with "S(U) B dot n")
        real_t B1n = 0.0;
        real_t B2n = 0.0;
        for (auto i=0;i<dim;i++) B1n += state1(i+1+dim) * nor(i);
        for (auto i=0;i<dim;i++) B2n += state2(i+1+dim) * nor(i);

        // Rusanov flux for the magnetic field normal component
        const real_t Bhatn = 0.5 * (B1n + B2n);
        const real_t Bjump1 = Bhatn - B1n;
        const real_t Bjump2 = Bhatn - B2n;

        source1.SetSize(gpSource.num_equations);
        source2.SetSize(gpSource.num_equations);
        source1(0) = 0.0;
        source2(0) = 0.0;
        for (auto i=0;i<dim;i++) source1(i+1) = -state1(i+1+dim) * Bjump1;
        for (auto i=0;i<dim;i++) source2(i+1) = +state2(i+1+dim) * Bjump2;
        for (auto i=0;i<dim;i++) source1(i+1+dim) = -state1(i+1) / state1(0) * Bjump1;
        for (auto i=0;i<dim;i++) source2(i+1+dim) = +state2(i+1) / state2(0) * Bjump2;
        source1(1+2*dim) = 0.0;
        source2(1+2*dim) = 0.0;
        for (auto i=0;i<dim;i++) source1(1+2*dim) -= (state1(i+1) / state1(0)) * state1(i+1+dim);
        for (auto i=0;i<dim;i++) source2(1+2*dim) += (state2(i+1) / state2(0)) * state2(i+1+dim);
        source1(1+2*dim) *= Bjump1;
        source2(1+2*dim) *= Bjump2;
    }

    void Grad(int side, const Vector &state1, const Vector &state2, const Vector &nor, FaceElementTransformations &Tr, DenseMatrix &grad) const override
    {MFEM_ABORT("Not implemented.");}
};
// We can also make other "numerical sources", such as the HLL version (recall Kailiang Wu's paper on PP MHD DG schemes).

// similar to HyperbolicFormIntegrator
class GPsourceFormIntegrator : public NonlinearFormIntegrator
{
private:
    const NumericalGPsource &numGPsource;
    const GPsource &gpsource;
    const int IntOrderOffset; // integration order offset, 2*p + IntOrderOffset.
    const real_t sign;
    #ifndef MFEM_THREAD_SAFE
    // Local storage for element integration
    Vector shape;              // shape function value at an integration point
    Vector state;              // state value at an integration point
    Vector source;             // GP source value at an integration point
    DenseTensor J;             // Jacobian matrix at an integration point
    DenseMatrix dshape;        // derivative of shape function at an integration point

    Vector shape1;  // shape function value at an integration point - first elem
    Vector shape2;  // shape function value at an integration point - second elem
    Vector state1;  // state value at an integration point - first elem
    Vector state2;  // state value at an integration point - second elem
    Vector nor;     // normal vector, see mfem::CalcOrtho()
    Vector source1; // source vector at the first element
    Vector source2; // source vector at the second element
    DenseMatrix JDotN;   // Ĵ(u±,x) n
    #endif

public:
    const int num_equations;  // the number of equations
    /**
     * @brief Construct a new GPsource Integrator object
     *
     * @param[in] numGPsource numerical GP source
     * @param[in] IntOrderOffset integration order offset
     * @param[in] sign sign of the convection term
     */
    GPsourceFormIntegrator(
       const NumericalGPsource &numGPsource_,
       const int IntOrderOffset_ = 0,
       const real_t sign_ = 1.):
        NonlinearFormIntegrator(),
        numGPsource(numGPsource_),
        gpsource(numGPsource_.GetSourceFunction()),
        IntOrderOffset(IntOrderOffset_),
        sign(sign_),
        num_equations(gpsource.num_equations) 
    {
        #ifndef MFEM_THREAD_SAFE
        state.SetSize(num_equations);
        source.SetSize(num_equations);
        state1.SetSize(num_equations);
        state2.SetSize(num_equations);
        source1.SetSize(num_equations);
        source2.SetSize(num_equations);
        JDotN.SetSize(num_equations);
        nor.SetSize(gpsource.dim);
        #endif
    }
 
    const GPsource &GetGPsource() { return gpsource; }
 
    /**
     * @brief Implements (-S(U)*divB, v) with left half computed by
     * GPsource::ComputeSource()
     *
     * @param[in] el local finite element
     * @param[in] Tr element transformation
     * @param[in] elfun local coefficient of basis
     * @param[out] elvect evaluated dual vector
     */
    void AssembleElementVector(const FiniteElement &el,
                               ElementTransformation &Tr,
                               const Vector &elfun, Vector &elvect) override
    {
        // current element's the number of degrees of freedom
        // does not consider the number of equations
        const auto dof = el.GetDof();
        
        #ifdef MFEM_THREAD_SAFE
        // Local storage for element integration

        // shape function value at an integration point
        Vector shape(dof);
        // derivative of shape function at an integration point
        DenseMatrix dshape(dof, Tr.GetSpaceDim());
        // divergence of vector shape function (ByNode) at an integration point
        // state value at an integration point
        Vector state(num_equations);
        // GP source value at an integration point
        Vector source(num_equations);
        #else
        // resize shape and gradient shape storage
        shape.SetSize(dof);
        dshape.SetSize(dof, Tr.GetSpaceDim());
        #endif

        // setDegree-up output vector
        elvect.SetSize(dof * num_equations);
        elvect = 0.0;

        // make state variable and output dual vector matrix form.
        const DenseMatrix elfun_mat(elfun.GetData(), dof, num_equations);
        DenseMatrix elvect_mat(elvect.GetData(), dof, num_equations); // make a view (not a copy)

        // obtain integration rule. If integration is rule is given, then use it.
        // Otherwise, get (2*p + IntOrderOffset) order integration rule
        const IntegrationRule *ir = IntRule;
        if (!ir)
        {
            const auto order = el.GetOrder()*2 + IntOrderOffset;
            ir = &IntRules.Get(Tr.GetGeometryType(), order);
        }

        // loop over integration points
        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            Tr.SetIntPoint(&ip);

            el.CalcShape(ip, shape);
            el.CalcPhysDShape(Tr, dshape);

            // compute current state value with given shape function values
            elfun_mat.MultTranspose(shape, state);

            // compute current divB value with given shape function gradient values
            const real_t divB = Vector(dshape.GetData(), dof*Tr.GetSpaceDim()) * Vector(elfun.GetData() + (1+Tr.GetSpaceDim())*dof, dof*Tr.GetSpaceDim());

            // compute source term value
            gpsource.ComputeSource(state, divB, Tr, source);
            // integrate (source, v) at the current quadrature point, and contribute to the final output
            AddMult_a_VWt(ip.weight * Tr.Weight() * sign, shape, source, elvect_mat);
        }
    }
 
    void AssembleElementGrad(const FiniteElement &el,
                             ElementTransformation &Tr,
                             const Vector &elfun, DenseMatrix &grad)
                             {MFEM_ABORT("Not implemented.");}
 
    /**
     * @brief Implements <-F̂(u⁻,u⁺,x) n, [v]> with abstract F̂ computed by
     * NumericalFlux::Eval() of the numerical flux object
     *
     * @param[in] el1 finite element of the first element
     * @param[in] el2 finite element of the second element
     * @param[in] Tr face element transformations
     * @param[in] elfun local coefficient of basis from both elements
     * @param[out] elvect evaluated dual vector \sum <source, v>
     */
    void AssembleFaceVector(const FiniteElement &el1,
                            const FiniteElement &el2,
                            FaceElementTransformations &Tr,
                            const Vector &elfun, Vector &elvect)
    {
        // current elements' the number of degrees of freedom does not consider the number of equations
        const int dof1 = el1.GetDof();
        const int dof2 = el2.GetDof();

        #ifdef MFEM_THREAD_SAFE
        // Local storage for element integration

        // shape function value at an integration point - first elem
        Vector shape1(dof1);
        // shape function value at an integration point - second elem
        Vector shape2(dof2);
        // normal vector (usually not a unit vector)
        Vector nor(Tr.GetSpaceDim());
        // state value at an integration point - first elem
        Vector state1(num_equations);
        // state value at an integration point - second elem
        Vector state2(num_equations);
        // numerical source on the faces
        Vector source1(num_equations);
        Vector source2(num_equations);
        #else
        shape1.SetSize(dof1);
        shape2.SetSize(dof2);
        #endif

        // overall DOFs for two elements
        elvect.SetSize((dof1 + dof2) * num_equations);
        elvect = 0.0;

        // setup views
        const DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equations);
        const DenseMatrix elfun2_mat(elfun.GetData() + dof1 * num_equations, dof2, num_equations);
        DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equations);
        DenseMatrix elvect2_mat(elvect.GetData() + dof1 * num_equations, dof2, num_equations);

        // Obtain integration rule. If integration is rule is given, then use it.
        // Otherwise, get (2*p + IntOrderOffset) order integration rule
        const IntegrationRule *ir = IntRule;
        if (!ir)
        {
            const int order = 2*std::max(el1.GetOrder(), el2.GetOrder()) + IntOrderOffset;
            ir = &IntRules.Get(Tr.GetGeometryType(), order);
        }
        // loop over integration points
        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);

            Tr.SetAllIntPoints(&ip); // set face and element int. points

            // Calculate basis functions on both elements at the face
            el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
            el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

            // Interpolate elfun at the point
            elfun1_mat.MultTranspose(shape1, state1);
            elfun2_mat.MultTranspose(shape2, state2);

            // Get the normal vector and the flux on the face
            if (nor.Size() == 1)  // if 1D, use 1 or -1.
            {
                // This assume the 1D integration point is in (0,1). This may not work
                // if this changes.
                nor(0) = (Tr.GetElement1IntPoint().x - 0.5) * 2.0;
            }
            else
            {
                CalcOrtho(Tr.Jacobian(), nor);
            }
            // Compute source1 and source2 using evaluated quantities
            numGPsource.Eval(state1, state2, nor, Tr, source1, source2);
            // We have adapted it so that only plus sign is needed below.

            // pre-multiply integration weight to flux
            AddMult_a_VWt(ip.weight*sign, shape1, source1, elvect1_mat);
            AddMult_a_VWt(ip.weight*sign, shape2, source2, elvect2_mat);
        }
    }
 
    void AssembleFaceGrad(const FiniteElement &el1,
                          const FiniteElement &el2,
                          FaceElementTransformations &Tr,
                          const Vector &elfun, DenseMatrix &elmat)
                          {MFEM_ABORT("Not implemented.");}
};



/// @brief Time dependent DG operator for hyperbolic conservation laws
class DG_MHD_GPsource : public TimeDependentOperator
{
private:
    const bool use_GPsource;

   const int num_equations; // the number of equations
   const int dim;
   FiniteElementSpace &vfes; // vector finite element space (ByNode)

   // Element integration form. Should contain ComputeFlux
   // only for setting up "nonlinearForm" and calling "GetMaxCharSpeed" (max speed obtained within computation)
   std::unique_ptr<HyperbolicFormIntegrator> hyperb_form_Int = nullptr;

   std::unique_ptr<GPsourceFormIntegrator> gpsource_form_Int = nullptr;

   // Base Nonlinear Form
   std::unique_ptr<NonlinearForm> nonlinearForm;

   // element-wise inverse mass matrix
   std::vector<DenseMatrix> invmass; // local scalar inverse mass

   // global maximum characteristic speed. Updated by form integrators
   mutable real_t max_char_speed;

   // auxiliary variable used in Mult
   mutable Vector z;

   // Compute element-wise inverse mass matrix
   void ComputeInvMass();

public:
   /**
    * @brief Construct a new DG_MHD_GPsource object
    *
    * @param vfes_ vector finite element space. Only tested for DG [Pₚ]ⁿ
    * @param formIntegrator_ integrator (F(u,x), grad v)
    */
   DG_MHD_GPsource(
      FiniteElementSpace &vfes_,
      std::unique_ptr<HyperbolicFormIntegrator> hyperb_form_Int_,
      std::unique_ptr<GPsourceFormIntegrator> gpsource_form_Int_,
      const bool use_GPsource_);
   /**
    * @brief Apply nonlinear form to obtain M⁻¹(DIVF + JUMP HAT(F))
    *
    * @param x current solution vector
    * @param y resulting dual vector to be used in an EXPLICIT solver
    */
   void Mult(const Vector &x, Vector &y) const override;
   // get global maximum characteristic speed to be used in CFL condition
   // where max_char_speed is updated during Mult.
   real_t GetMaxCharSpeed() { return max_char_speed; }
   void Update();

};

//////////////////////////////////////////////////////////////////
///        MHD with Godunov-Powell term IMPLEMENTATION         ///
//////////////////////////////////////////////////////////////////

// Implementation of class DG_MHD_GPsource
DG_MHD_GPsource::DG_MHD_GPsource(
   FiniteElementSpace &vfes_,
   std::unique_ptr<HyperbolicFormIntegrator> hyperb_form_Int_,
   std::unique_ptr<GPsourceFormIntegrator> gpsource_form_Int_,
   const bool use_GPsource_)
   : TimeDependentOperator(vfes_.GetTrueVSize()),
     num_equations(hyperb_form_Int_->num_equations),
     dim(vfes_.GetMesh()->SpaceDimension()),
     vfes(vfes_),
     hyperb_form_Int(std::move(hyperb_form_Int_)),
     gpsource_form_Int(std::move(gpsource_form_Int_)),
     use_GPsource(use_GPsource_),
     z(vfes_.GetTrueVSize())
{
   // Standard local assembly and inversion for energy mass matrices.
   ComputeInvMass();

#ifndef MFEM_USE_MPI
   nonlinearForm.reset(new NonlinearForm(&vfes));
#else
   ParFiniteElementSpace *pvfes = dynamic_cast<ParFiniteElementSpace *>(&vfes);
   if (pvfes)
   {
      nonlinearForm.reset(new ParNonlinearForm(pvfes));
   }
   else
   {
      nonlinearForm.reset(new NonlinearForm(&vfes));
   }
#endif
   /** For nonlinear operators, the "matrix" assembly levels usually do not make
       sense, so only PARTIAL and NONE (matrix-free) are supported. */
    
   // Add the integrators to the nonlinear form
   nonlinearForm->AddDomainIntegrator(hyperb_form_Int.get());
   nonlinearForm->AddInteriorFaceIntegrator(hyperb_form_Int.get()); // only interior faces

   if (use_GPsource){
    // Add the Godunov-Powell source term
        nonlinearForm->AddDomainIntegrator(gpsource_form_Int.get());
        nonlinearForm->AddInteriorFaceIntegrator(gpsource_form_Int.get()); // only interior faces
   }

   nonlinearForm->UseExternalIntegrators(); // indicate that integrators are not owned by the NonlinearForm

}

void DG_MHD_GPsource::ComputeInvMass()
{
   InverseIntegrator inv_mass(new MassIntegrator());

   invmass.resize(vfes.GetNE());
   for (auto i=0; i<vfes.GetNE(); i++)
   {
      const int dof = vfes.GetFE(i)->GetDof();
      invmass[i].SetSize(dof);
      inv_mass.AssembleElementMatrix(*vfes.GetFE(i),
                                     *vfes.GetElementTransformation(i),
                                     invmass[i]);
   }
}


void DG_MHD_GPsource::Mult(const Vector &x, Vector &y) const
{
    // 0. Reset wavespeed computation before operator application.
    hyperb_form_Int->ResetMaxCharSpeed();

    // 1. Apply Nonlinear form to obtain an auxiliary result
    nonlinearForm->Mult(x, z);

    // 2. Apply Linear form
    // Apply block inverse mass
    Vector zval; // z_loc, dof*num_eq
    DenseMatrix current_zmat; // view of element auxiliary result, dof x num_eq
    DenseMatrix current_ymat; // view of element result, dof x num_eq
    Array<int> vdofs;
    for (auto i=0; i<vfes.GetNE(); i++)
    {
        const auto dof = vfes.GetFE(i)->GetDof();
        // ALWAYS returns ByNodes ordering (like separate fields)
        vfes.GetElementVDofs(i, vdofs);
        // copy from the global vector to the local vector
        z.GetSubVector(vdofs, zval);
        // each column is a field and each row is a DOF
        current_zmat.UseExternalData(zval.GetData(), dof, num_equations);
        current_ymat.SetSize(dof, num_equations);
        // invmass left multiplies the auxiliary result
        mfem::Mult(invmass[i], current_zmat, current_ymat);
        // return the result to the global vector
        y.SetSubVector(vdofs, current_ymat.GetData());
    }
    max_char_speed = hyperb_form_Int->GetMaxCharSpeed();
}

void DG_MHD_GPsource::Update()
{
   nonlinearForm->Update();
   height = nonlinearForm->Height();
   width = height;
   z.SetSize(height);

   ComputeInvMass();
}


// triangular mesh for [0,1]^2 with periodic BC
Mesh MHDMesh(const int problem)
{
    return Mesh("../data/periodic-square-tri.msh");
}

// Initial condition
VectorFunctionCoefficient MHDInitialCondition(const int problem,
                                              real_t specific_heat_ratio)
{
   switch (problem)
   {
    case 1: // Orzag-Tang vortex (with 3 refines, negative density emerge at t=0.23)
        return VectorFunctionCoefficient(6, [specific_heat_ratio](const Vector &x,Vector &y)
            {
            MFEM_ASSERT(x.Size() == 2, "");
            const real_t density = 25.0/(36.0*M_PI);
            const real_t velocity_x = -std::sin(2*M_PI*x(1));
            const real_t velocity_y = std::sin(2*M_PI*x(0));
            const real_t magnetic_x = -std::sin(2*M_PI*x(1))/std::sqrt(4*M_PI);
            const real_t magnetic_y = std::sin(4*M_PI*x(0))/std::sqrt(4*M_PI);
            const real_t pressure = 5.0/(12.0*M_PI);
            const real_t energy =
            pressure / (specific_heat_ratio - 1.0) +
            0.5 * (density * (velocity_x * velocity_x + velocity_y * velocity_y) + (magnetic_x*magnetic_x + magnetic_y*magnetic_y));

            y(0) = density;
            y(1) = density * velocity_x;
            y(2) = density * velocity_y;
            y(3) = magnetic_x;
            y(4) = magnetic_y;
            y(5) = energy;
            });
      default:
         MFEM_ABORT("Problem Undefined");
   }
}

} // namespace mfem
