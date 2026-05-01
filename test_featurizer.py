import pandas as pd

class Featurizer:

    def name(self):
        return 'count transactions'

    def featurize(
        self,
        cdr=None,
        mobile_money=None,
        mobile_data=None,
        recharges=None,
        antennas=None,
        shapefiles=None
    ):
        outgoing = cdr.copy()
        outgoing.rename(
            columns={
                'caller_msisdn': 'ego',
                'recipient_msisdn': 'alter',
            },
            inplace=True
        )
        outgoing['direction'] = 'outgoing'

        incoming = cdr.copy()
        incoming.rename(
            columns={
                'caller_msisdn': 'alter',
                'recipient_msisdn': 'ego',
            },
            inplace=True
        )
        incoming['direction'] = 'incoming'

        bidirectional = pd.concat((outgoing, incoming))

        num_outgoing = outgoing.groupby('ego').apply(len).rename('num_outgoing')
        num_incoming = incoming.groupby('ego').apply(len).rename('num_incoming')
        return num_incoming.to_frame().join(num_outgoing)