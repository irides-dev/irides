"""
-------------------------------------------------------------------------------

Common tooling for filter-signature unit tests.

-------------------------------------------------------------------------------
"""

from irides.tools import impulse_response_tools as ir_tools


def make_impulse_response(this, filter_order):
    """Local helper that builds h[n] consistently"""

    # detect if the filter is mbox type
    is_mbox = this.design.design_id().find("mbox") > -1

    # build impulse response
    n_start = 0
    mu_min = this.design.minimum_mu_value(filter_order)
    mu_trg = 20 * mu_min
    if is_mbox:
        mu_trg = this.design.adjust_mu_to_ensure_integer_length_box(
            mu_trg, filter_order
        )
    n_end = 50 * mu_trg

    ds = this.f_sig.generate_impulse_response(
        n_start, n_end, mu_trg, filter_order
    )

    return ds, mu_trg


# noinspection PyPep8Naming
def common_impulse_response_moment_tests(
    this, precisions: dict, proportional=False
):
    """Tests that M0, M1, and M2 are correct to within set precision"""

    # filter orders and moments
    filter_orders = this.design.get_valid_filter_orders(strict=False)
    moments = ir_tools.get_valid_moments()

    # defs
    moment_cols = ["M0", "M1", "M2"]

    # iter over filter orders
    for i, filter_order in enumerate(filter_orders):

        # make impulse response
        ds, mu_trg = make_impulse_response(this, filter_order)

        # moments
        moments_test = [
            ir_tools.calculate_impulse_response_moment(ds.to_ndarray(), moment)
            for moment in moments
        ]
        moments_design = this.design.moment_values(mu_trg, filter_order)

        # tests
        for j, moment in enumerate(moments):

            with this.subTest(
                msg="order: {0}, M: {1}".format(filter_order, moment)
            ):

                precision = precisions[moment_cols[j]]

                if proportional:
                    this.assertAlmostEqual(
                        moments_test[j] / moments_design[j], 1.0, precision
                    )
                else:
                    this.assertAlmostEqual(
                        moments_test[j], moments_design[j], precision
                    )


# noinspection PyPep8Naming
def common_impulse_response_initial_values_tests(this, precisions: dict):
    """Tests that h[0], h[1], and h[2] are correct to within set precision"""

    # filter orders and moments
    filter_orders = this.design.get_valid_filter_orders(strict=False)

    # iter over filter orders
    for i, filter_order in enumerate(filter_orders):

        # make impulse response
        ds, mu_trg = make_impulse_response(this, filter_order)

        # h[0], h[1], h[2] values
        hns_test = ds.v_axis[:3]
        hns_design = this.design.initial_hn_values(mu_trg, filter_order)

        # tests
        for j in range(hns_test.shape[0]):

            with this.subTest(
                msg="order: {0}, h[{1}] test".format(filter_order, j)
            ):

                this.assertAlmostEqual(
                    hns_test[j] / hns_design[j], 1.0, precisions["hk"]
                )
