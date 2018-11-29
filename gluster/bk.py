
    def __zero_new_centers(self):
        if self.module.bias:
            self._zero_new_centers_bias()
        if self.module.weight:
            self._zero_new_centers_weight()

    def __accum_new_centers(self, assign_i):
        if self.module.bias:
            self._accum_new_centers_bias(assign_i)
        if self.module.weight:
            self._accum_new_centers_weight(assign_i)

    def __update_batch(self, cluster_size):
        if self.module.bias:
            self._update_batch_bias(cluster_size)
        if self.module.weight:
            self._update_batch_weight(cluster_size)

    def __update_online(self, pre_size, new_size):
        if self.module.bias:
            self._update_online_bias(pre_size, new_size)
        if self.module.weight:
            self._update_online_weight(pre_size, new_size)


    def update_batch(self, assign_i):
        """Update Clusters
        E: update C
        E: C_c = mean_i(g_i * I(c==a_i))
        """
        for module in self.G.keys():
            self.G[module].update_batch(assign_i)

    def update_online(self, assign_i):
        """Update Clusters
        E: update C
        E: C_c = mean_i(g_i * I(c==a_i))
        """
        for module in self.G.keys():
            self.G[module].update_online(assign_i)

    def normalize(self):
        for module in self.G.keys():
            self.G[module].normalize()


class GradientClusterOnlineMemory(GradientCluster):
    """
    Main issue: when should we reset the cluster size?
    """
    def __init__(self, model, min_size, train_size, **kwargs):
        super(GradientClusterOnlineMemory, self).__init__(model, **kwargs)
        self.train_size = train_size
        self.assign_i = torch.zeros(train_size, 1).long().to(self.device)

    def em_step(self):
        if self.is_active:
            assign_i = self.assign()
            if not self.is_eval:
                self.update(assign_i)
            return assign_i

    def assign(self, idx):
        """Assign inputs to clusters
        M: assign input
        M: a_i = argmin_c(|g_i-C_c|^2)
        = argmin_c gg+CC-2*gC
        = argmin_c CC-2gC
        """
        batch_dist = self.G.get_dist()
        total_dist = torch.stack(batch_dist).sum(0)
        batch_dist, assign_i = total_dist.min(0)
        assign_i = assign_i.unsqueeze(1)
        batch_dist = batch_dist.unsqueeze(1)
        return assign_i, batch_dist

    def update(self, assign_i, batch_dist, idx):
        # E: update C
        # E: C_c = mean_i(g_i * I(c==a_i))
        # Keep a full internal assignment list
        self.assign_i[idx] = assign_i
        assign_i = assign_i.unsqueeze(1)

        # Keep an exact cluster size given the internal assignment
        # TODO: is this the only place we use self.assign_i?
        self.cluster_size.fill_(0)
        self.cluster_size.scatter_add_(0, assign_i,
                                       torch.ones_like(assign_i).float())
        counts = torch.zeros(self.nclusters, 1).cuda()
        counts.scatter_add_(0, assign_i, torch.ones(assign_i.shape).cuda())
        pre_size = self.cluster_size-counts

        # Reinit if cluster size small
        reinits = (self.cluster_size < self.min_size)
        nreinits = reinits.sum()
        self.reinits += reinits.long()

        for module in self.inputs.keys():
            Ai = self.inputs[module]
            Go = self.ograds[module]

            # TODO: refactor copies of this
            # bias
            Cb = self.centers[module.bias]
            Cb_new = torch.zeros_like(Cb)
            Cb_new.scatter_add_(0, assign_i.expand_as(Go), Go)
            Cb.mul_(pre_size).add_(Cb_new).div_(self.cluster_size)

            # weight
            Ci, Co = self.centers[module.weight]
            Ci_new = torch.zeros_like(Ci)
            Co_new = torch.zeros_like(Co)
            # TODO: SVD
            Ci_new.scatter_add_(0, assign_i.expand_as(Ai), Ai)
            Co_new.scatter_add_(0, assign_i.expand_as(Go), Go)
            # Update clusters using the size
            Ci.mul_(pre_size).add_(Ci_new).div_(self.cluster_size)
            Co.mul_(pre_size).add_(Co_new).div_(self.cluster_size)

            # reinit centers
            if nreinits > 0:
                # TODO: ?reinit from the largest cluster?
                pass

