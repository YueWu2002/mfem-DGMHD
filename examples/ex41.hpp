//                  MFEM Example 18 - Serial/Parallel Shared Code for MHD equations
//                      (Implementation of Time-dependent DG Operator)
//
// This code provide example problems for the MHD equations and implements
// the time-dependent DG operator given by the equation:
//
//            (u_t, v)_T - (F(u), ∇ v)_T + <F̂(u, n), [[v]]>_F = 0.
//
// This operator is designed for explicit time stepping methods. Specifically,
// the function DGHyperbolicConservationLaws::Mult implements the following
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
// details, refer to the documentation of DGHyperbolicConservationLaws::Mult.
//

#include <functional>
#include "mfem.hpp"

namespace mfem
{

/// @brief Time dependent DG operator for hyperbolic conservation laws
class DGHyperbolicConservationLaws : public TimeDependentOperator
{
private:
   const int num_equations; // the number of equations
   const int dim;
   FiniteElementSpace &vfes; // vector finite element space
   // Element integration form. Should contain ComputeFlux
   std::unique_ptr<HyperbolicFormIntegrator> formIntegrator;
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
    * @brief Construct a new DGHyperbolicConservationLaws object
    *
    * @param vfes_ vector finite element space. Only tested for DG [Pₚ]ⁿ
    * @param formIntegrator_ integrator (F(u,x), grad v)
    */
   DGHyperbolicConservationLaws(
      FiniteElementSpace &vfes_,
      std::unique_ptr<HyperbolicFormIntegrator> formIntegrator_);
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
///        HYPERBOLIC CONSERVATION LAWS IMPLEMENTATION         ///
//////////////////////////////////////////////////////////////////

// Implementation of class DGHyperbolicConservationLaws
DGHyperbolicConservationLaws::DGHyperbolicConservationLaws(
   FiniteElementSpace &vfes_,
   std::unique_ptr<HyperbolicFormIntegrator> formIntegrator_)
   : TimeDependentOperator(vfes_.GetTrueVSize()),
     num_equations(formIntegrator_->num_equations),
     dim(vfes_.GetMesh()->SpaceDimension()),
     vfes(vfes_),
     formIntegrator(std::move(formIntegrator_)),
     z(vfes_.GetTrueVSize())
{
   // Standard local assembly and inversion for energy mass matrices.
   ComputeInvMass();

   /** For nonlinear operators, the "matrix" assembly levels usually do not make
       sense, so only PARTIAL and NONE (matrix-free) are supported. */
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
   nonlinearForm->AddDomainIntegrator(formIntegrator.get());
   nonlinearForm->AddInteriorFaceIntegrator(formIntegrator.get()); // only interior faces
   nonlinearForm->UseExternalIntegrators(); // indicate that integrators are not owned by the NonlinearForm

}

void DGHyperbolicConservationLaws::ComputeInvMass()
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


void DGHyperbolicConservationLaws::Mult(const Vector &x, Vector &y) const
{
    // 0. Reset wavespeed computation before operator application.
    formIntegrator->ResetMaxCharSpeed();

    // 1. Apply Nonlinear form to obtain an auxiliary result
    //         z = - <F̂(u_h,n), [[v]]>_e + (F(u_h), ∇v)
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
    max_char_speed = formIntegrator->GetMaxCharSpeed();
}

void DGHyperbolicConservationLaws::Update()
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
