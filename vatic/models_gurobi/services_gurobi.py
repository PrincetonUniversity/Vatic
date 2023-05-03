def storage_services(model):
    # No need to run the whole function below when no storage unit exits
    if len(model._Storage) == 0:
        return model
    else:
        raise Exception('Storage Units are in the assets. Need to rewrite the original storage_services() function to add constrs')


def ancillary_services(model):
    '''
    Defines ancillary services: regulation, spinning reserve, nonspinning reserve, operational reserve, flexible ramp
    ## NOTE: As in most markets, the value of ancillary services from high to low is regulation, spinning reserve, nonspinning reserve, and supplemental reserve.
    ##       We allow for a higher-quality ancillary service to be subtituted for a lower-quality one
    ##       Flexible ramp is treated differently, again as it is in most markets. There is no bid for flexible ramp, and it is priced at opportunity cost
    '''
    md = model._model_data

    system = md._data['system']
    elements = md._data['elements']

    ## list of possible ancillary services coming
    ## from model_data
    ancillary_service_list = ['spinning_reserve_requirement',
                              'non_spinning_reserve_requirement',
                              'regulation_up_requirement',
                              'regulation_down_requirement',
                              'supplemental_reserve_requirement',
                              'flexible_ramp_up_requirement',
                              'flexible_ramp_down_requirement',
                              ]

    if 'zone' not in elements:
        elements['zone'] = dict()
    if 'area' not in elements:
        elements['area'] = dict()

    ## check and see if each one of these services appears anywhere in model_data
    def _check_for_requirement(requirement):
        if requirement in system:
            return True
        for zone in elements['zone'].values():
            if requirement in zone:
                return True
        for area in elements['area'].values():
            if requirement in area:
                return True
        return False

    ## flags for if ancillary services appear
    add_spinning_reserve = _check_for_requirement(
        'spinning_reserve_requirement')
    add_non_spinning_reserve = _check_for_requirement(
        'non_spinning_reserve_requirement')
    add_regulation_reserve = (
                _check_for_requirement('regulation_up_requirement') or
                _check_for_requirement('regulation_down_requirement'))
    add_supplemental_reserve = _check_for_requirement(
        'supplemental_reserve_requirement')
    add_flexi_ramp_reserve = (
                _check_for_requirement('flexible_ramp_up_requirement') or
                _check_for_requirement('flexible_ramp_down_requirement'))

    ## check here and break if there's nothing to do
    no_reserves = not (
                add_spinning_reserve or add_non_spinning_reserve or add_regulation_reserve or add_supplemental_reserve or add_flexi_ramp_reserve)

    ## add a flag for which brach we took here
    if no_reserves:
        model._nonbasic_reserves = False
        model._regulation_service = None
        model._spinning_reserve = None
        model._non_spinning_reserve = None
        model._supplemental_reserve = None
        model._flexible_ramping = None
        return model
    else:
        raise Exception(
            'Ancillary services are there. Need to rewrite the original ancillary_services() function to add constrs')
