class ValParams():
    def __init__(self):

                self.total_num=0
                self.flare_id=0
                self.micro_id=0
                self.norm_num=0
                self.norm_id=0
                self.run_id=0
                self.elps=1e-6
              
                #p4-cpg
                self.flare_num=0
                self.micro_num=0
                self.g_num=0
                self.g_end=0
                self.c_num=0
                self.c_start=0
                self.p_num=0
                self.p_start=0
                self.p_end=0
                
                
                self.phase_c=0
                self.phase_p=0
                self.phase_g=0
                #p4

                self.phase_confused_metrix = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
                
                # valid for whole sequence
                self.labellist=[]
                self.predictionlist=[]
                self.confidencelist=[]
                self.eventprelist=[]
                self.ranklist=[]


                self.data_groups=[]

                