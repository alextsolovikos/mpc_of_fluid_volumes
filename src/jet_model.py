import numpy as np
import argparse
import data_loader





if __name__ == '__name__':

    # Input arguments
    parser = argparse.ArgumentParser(description = 'Compute a DMDcsp model of the vertical velocity component in the field inside the control grid.')
    parser.add_argument('--train', default=False, action='store_true', help='Flag to train a new DMDcsp model.')
    parser.add_argument('--training_data_dir', dest='training_data_dir', help='Training snapshot data location.')
    parser.add_argument('--test_data_dir', dest='test_data_dir', help='Test snapshot data location.')
    parser.add_argument('--model', dest='model_name', help='Model pickle object location.')
    args = parser.parse_args()
    

    # Load snapshot data
    snapshot_data = data_loader.SnapshotData(args.training_data_dir, start=0, end=50)

    # Train or Load DMDcsp model
    if args.train:
        # Train model
        q = 40
        Y0 = snapshot_data.Y0
        Y1 = snapshot_data.Y1
        U0 = snapshot_data.U0

        # Full model
        model = DMDcsp.DMDcsp(Y0, Y1, U0, q=q)

        # Sparse model
        num = 50
        n_iter = 5
        gamma = np.logspace(2., 7., num=num)
        stats = model.sparse_batch(gamma, n_iter)

        order = stats['nx']
        Ploss = stats['P_loss']
        z0 = stats['z_0']

        # Choose model
        sys_i = int(input('Choose the sparse model id to use: '))

        nx = stats['nx'][sys_i]

        print('\nRank of observability matrix: %d of %d' % (np.linalg.matrix_rank(control.obsv(model.rsys[sys_i].A, C)), nx))

        Ts = 10
        print('Mode frequencies:')
        print(np.abs(np.angle(np.diag(model.sys_eig[sys_i])))/(2.0*np.pi*Ts))

        # Save full model & final model
        pickle.dump(model, open('data/' + args.model_name + '_full.p', 'wb'))

        final_model = [model.rsys[sys_i], C, Qe, Re, sens]
        pickle.dump(final_model, open('data/' + args.model_name + '_sparse.p', 'wb'))
        

    else:
        # Load existing model
        model = pickle.load(open('data/' + args.model_name + '_full.p', 'rb'))



    """ Plots """

    order = model.sp_stats['nx']
    Ploss = model.sp_stats['P_loss']
    z0 = model.sp_stats['z_0']
    nx = sys.nx

    order_uq, indices = np.unique(order, return_index=True)
    Ploss_uq = Ploss[indices]


    # Plot percentage loss
    fig, axs = plt.subplots(1, figsize=(6,4), facecolor='w', edgecolor='k')
    plt.subplots_adjust(hspace=0.6, left=0.18, right=0.95, top=0.95, bottom=0.18)

    axs.plot(order_uq, Ploss_uq, c='k', zorder=9, clip_on=False)
    axs.scatter(order_uq, Ploss_uq, s=m_size, marker='o', facecolor='none', edgecolor='k', zorder=10, clip_on=False)
    axs.scatter(order[sys_i], Ploss[sys_i], s=spm_size, marker='x', color='darkred', zorder=11, clip_on=False)

    axs.set_axisbelow(True)
    axs.set_xlabel('$n_x$')
    axs.set_ylabel('$P_{\\mathrm{error}},\ \%$')
    axs.set_xticks(np.arange(0, 31, 5))
    axs.set_yticks(np.arange(0, 101, 20))
    plt.grid(True)
    axs.set_xlim([0,30])
    axs.set_ylim([0,100])
    plt.savefig('/Users/atsol/research/papers/dmdcsp-paper/figures/Ploss.eps')


    # Plot eigenvalues
    lamb = np.diag(model.Lambda)
    lamb_sp = model.sys_eig[sys_i]
    print('eigenvalue magnitudes:')
    print(np.abs(lamb_sp))
    fig, axs = plt.subplots(1, figsize=(6,4), facecolor='w', edgecolor='k')
    plt.subplots_adjust(hspace=0.6, left=0.22, right=0.95, top=0.95, bottom=0.18)

    # Unit circle
    circle = Circle((0,0), 1.0, edgecolor='k', facecolor='none', linewidth=1)
    axs.add_artist(circle)

    # Eigenvalues of full system
    axs.scatter(np.real(lamb), np.imag(lamb), s=m_size, marker='o', facecolor='none', edgecolor='k', zorder=15, clip_on=False)

    # Eigenvalues of sparse system
    axs.scatter(np.real(lamb_sp), np.imag(lamb_sp), s=spm_size, marker='x', color='darkred', zorder=16, clip_on=False)

    axs.set_axisbelow(True)
    axs.set_xlabel('$\mathrm{Re}(\lambda_i)$')
    axs.set_ylabel('$\mathrm{Im}(\lambda_i)$')
    axs.set_xticks(np.arange(0.97, 1.02, 0.01))
    axs.set_yticks(np.arange(-0.2, 0.25, 0.1))
    axs.set_xlim([0.97, 1.005])
    axs.set_ylim([-0.25, 0.25])
    plt.grid(True)
    plt.savefig('/Users/atsol/research/papers/dmdcsp-paper/figures/eigenvalues.eps')


    # Plot frequencies

    Ts = 5.0 * 0.00125

    # Full system
    freq = np.abs(np.imag(np.log(lamb))/(2.0*np.pi)/Ts)
    ampl = np.abs(z0[0])

    # Sparse system
    freq_sp = np.abs(np.imag(np.log(lamb_sp))/(2.0*np.pi)/Ts)
    ampl_sp = np.abs(z0[sys_i][:nx])
    print(freq_sp)
    print(ampl_sp)

    fig, axs = plt.subplots(1, figsize=(6,4), facecolor='w', edgecolor='k')
    plt.subplots_adjust(hspace=0.6, left=0.22, right=0.95, top=0.95, bottom=0.2)

    axs.scatter(freq, ampl, s=m_size, marker='o', facecolor='none', edgecolor='k', zorder=10, clip_on=False)
    axs.scatter(freq_sp, ampl_sp, s=spm_size, marker='x', color='darkred', zorder=11, clip_on=False)
    axs.set_axisbelow(True)
    axs.set_xlabel('$\mathrm{Im}(\log(\lambda_i))$')
    axs.set_ylabel('$\|x(0)\|$')
    axs.set_xlim([0, 6])
    axs.set_ylim([0, 100])
    axs.set_xticks(np.arange(0, 6.5, 1.0))
    axs.set_yticks(np.arange(0, 101, 20))
    plt.grid(True)
    plt.savefig('/Users/atsol/research/papers/dmdcsp-paper/figures/amplitudes.eps')






