import itertools
import copy
import numpy

from mindquantum.core.parameterresolver import ParameterResolver


def uccsd_singlet_generator(n_virtual, n_occupied, irrep_id, mo_occ, anti_hermitian=True):

    
    from mindquantum.core.operators import (  
        FermionOperator,
    )
    from mindquantum.core.operators.utils import down_index, up_index

    # Initialize operator
    generator = FermionOperator()

    # Generate excitations
    spin_index_functions = [up_index, down_index]

    # 计算基态分子整体的不可约表示
    mol_sym_id = 0
    in_mo_occ = mo_occ.astype(int)
    for i in range(len(in_mo_occ)):
        for j in range(in_mo_occ[i]):
            mol_sym_id = mol_sym_id ^ irrep_id[i]

    # Generate all spin-conserving single and double excitations derived from one spatial occupied-virtual pair
    for i, (p, q) in enumerate(itertools.product(range(n_virtual), range(n_occupied))):
        # Get indices of spatial orbitals
        virtual_spatial = n_occupied + p
        occupied_spatial = q
        # 计算单激发轨道的不可约表示
        temp_mo_occ = copy.deepcopy(in_mo_occ)
        temp_mo_occ[occupied_spatial] -= 1
        temp_mo_occ[virtual_spatial] += 1
        temp_mol_sym_id = 0
        flag_singles = False
        for m in range(len(temp_mo_occ)):
            for j in range(temp_mo_occ[m]):
                temp_mol_sym_id = temp_mol_sym_id ^irrep_id[m]
        if(temp_mol_sym_id == mol_sym_id):
            flag_singles = True
        # 计算双激发轨道的不可约表示
        temp_mo_occ = copy.deepcopy(in_mo_occ)
        temp_mo_occ[occupied_spatial] -= 2
        temp_mo_occ[virtual_spatial] += 2
        temp_mol_sym_id = 0
        flag_doubles = False
        for m in range(len(temp_mo_occ)):
            for j in range(temp_mo_occ[m]):
                temp_mol_sym_id = temp_mol_sym_id ^irrep_id[m]
        if(temp_mol_sym_id == mol_sym_id):
            flag_doubles = True
        for spin in range(2):
            # Get the functions which map a spatial orbital index to a
            # spin orbital index
            this_index = spin_index_functions[spin]
            other_index = spin_index_functions[1 - spin]

            # Get indices of spin orbitals
            virtual_this = this_index(virtual_spatial)
            virtual_other = other_index(virtual_spatial)
            occupied_this = this_index(occupied_spatial)
            occupied_other = other_index(occupied_spatial)

            # Generate single excitations
            if flag_singles:
                coeff = ParameterResolver({f's_{i}': 1})
                generator += FermionOperator(((virtual_this, 1), (occupied_this, 0)), coeff)
                if anti_hermitian:
                    generator += FermionOperator(((occupied_this, 1), (virtual_this, 0)), -1 * coeff)

            # Generate double excitation
            if flag_doubles:
                coeff = ParameterResolver({f'd1_{i}': 1})
                generator += FermionOperator(
                    ((virtual_this, 1), (occupied_this, 0), (virtual_other, 1), (occupied_other, 0)), coeff
                )
                if anti_hermitian:
                    generator += FermionOperator(
                        ((occupied_other, 1), (virtual_other, 0), (occupied_this, 1), (virtual_this, 0)), -1 * coeff
                    )

    # Generate all spin-conserving double excitations derived from two spatial occupied-virtual pairs
    for i, ((p, q), (r, s)) in enumerate(
        itertools.combinations(itertools.product(range(n_virtual), range(n_occupied)), 2)
    ):
        # Get indices of spatial orbitals
        virtual_spatial_1 = n_occupied + p
        occupied_spatial_1 = q
        virtual_spatial_2 = n_occupied + r
        occupied_spatial_2 = s
        # 计算双激发轨道的不可约表示
        temp_mo_occ = copy.deepcopy(in_mo_occ)
        temp_mo_occ[occupied_spatial_1] -= 1
        temp_mo_occ[occupied_spatial_2] -= 1
        temp_mo_occ[virtual_spatial_1] += 1
        temp_mo_occ[virtual_spatial_2] += 1
        temp_mol_sym_id = 0
        for m in range(len(temp_mo_occ)):
            for j in range(temp_mo_occ[m]):
                temp_mol_sym_id = temp_mol_sym_id ^irrep_id[m]
        if(temp_mol_sym_id != mol_sym_id):
            continue
        # Generate double excitations
        coeff = ParameterResolver({f'd2_{i}': 1})
        for (spin_a, spin_b) in itertools.product(range(2), repeat=2):
            # Get the functions which map a spatial orbital index to a
            # spin orbital index
            index_a = spin_index_functions[spin_a]
            index_b = spin_index_functions[spin_b]

            # Get indices of spin orbitals
            virtual_1_a = index_a(virtual_spatial_1)
            occupied_1_a = index_a(occupied_spatial_1)
            virtual_2_b = index_b(virtual_spatial_2)
            occupied_2_b = index_b(occupied_spatial_2)

            if virtual_1_a == virtual_2_b or occupied_1_a == occupied_2_b:
                continue

            generator += FermionOperator(
                ((virtual_1_a, 1), (occupied_1_a, 0), (virtual_2_b, 1), (occupied_2_b, 0)), coeff
            )
            if anti_hermitian:
                generator += FermionOperator(
                    ((occupied_2_b, 1), (virtual_2_b, 0), (occupied_1_a, 1), (virtual_1_a, 0)), -1 * coeff
                )

    return generator
