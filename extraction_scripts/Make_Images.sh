#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/icetray-start
#METAPROJECT combo/stable 

#python  /data/user/dpankova/double_pulse/Make_Images.py -o Data_TEST_129500 -i /data/exp/IceCube/2017/filtered/level2pass2a/0512/Run00129500/Level2pass2_IC86.2016_data_Run00129500_Subrun00000000_00000190.i3.zst -gcd /data/exp/IceCube/2017/filtered/level2pass2a/0512/Run00129500/Level2pass2_IC86.2016_data_Run00129500_0512_90_312_GCD.i3.zst -t data -set 00129500

python /data/user/dpankova/double_pulse/Make_Images.py -o Data_TEST -i /data/exp/IceCube/2016/filtered/level2/1209/Run00128900/Level2_IC86.2016_data_Run00128900_Subrun0000001*.i3.bz2 -gcd /data/exp/IceCube/2016/filtered/level2/1209/Run00128900/Level2_IC86.2016_data_Run00128900_1209_51_284_GCD.i3.gz -t data -set 00128900

#python  /data/user/dpankova/double_pulse/Make_Images.py -o MuonGun_Make_image_TEST -i  /data/sim/IceCube/2016/filtered/level2/MuonGun/21315/0000000-0000999/Level2_IC86.2016_MuonGun.021315.00000*.i3.zst  -gcd /cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz -t muongun -set 21315  -y 2016

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/9255/00000-00999/Level2_IC86.2011_corsika.009255.000010.i3.bz2 -gcd /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/9255/01000-01999/G*.i3.gz -t corsika -set 9255 -y 2011  

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/10309/00000-00999/Level2_IC86.2011_corsika.010309.000010.i3.bz2 -gcd /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/10309/01000-01999/G*.i3.gz -t corsika -set 10309 -y 2011  

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/10475/00000-00999/Level2_IC86.2011_corsika.010475.00001*.i3.bz2 -gcd /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/10475/01000-01999/G*.i3.gz -t corsika -set 10475 -y 2011  

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/10651/00000-00999/Level2_IC86.2011_corsika.010651.00001*.i3.bz2 -gcd /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/10651/01000-01999/G*.i3.gz -t corsika -set 10651 -y 2011  

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/10668/00000-00999/Level2_IC86.2011_corsika.010668.00001*.i3.bz2 -gcd /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/10668/01000-01999/G*.i3.gz -t corsika -set 10668 -y 2011  

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/10899/00000-00999/Level2_IC86.2011_corsika.010899.00001*.i3.bz2 -gcd /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/10899/01000-01999/G*.i3.gz -t corsika -set 10899 -y 2011  

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/10784/00000-00999/Level2_IC86.2011_corsika.010784.00001*.i3.bz2 -gcd /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/10309/01000-01999/G*.i3.gz -t corsika -set 10784 -y 2011  

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/10281/00000-00999/Level2_IC86.2011_corsika.010281.00001*.i3.bz2 -gcd /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/10309/01000-01999/G*.i3.gz -t corsika -set 10281 -y 2011  

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/10282/00000-00999/Level2_IC86.2011_corsika.010282.00001*.i3.bz2 -gcd /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/10309/01000-01999/G*.i3.gz -t corsika -set 10282 -y 2011  

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/9036/00000-00999/Level2_IC86.2011_corsika.009036.00001*.i3.bz2 -gcd /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/10309/01000-01999/G*.i3.gz -t corsika -set 9036 -y 2011  

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/9255/00000-00999/Level2_IC86.2011_corsika.009255.00001*.i3.bz2 -gcd /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/10309/01000-01999/G*.i3.gz -t corsika -set 9255 -y 2011  

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/9622/00000-00999/Level2_IC86.2011_corsika.009622.00001*.i3.bz2 -gcd /data/sim/IceCube/2011/filtered/level2/CORSIKA-in-ice/10309/01000-01999/G*.i3.gz -t corsika -set 9622 -y 2011  

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/12379/0000000-0000999/Level2_IC86.2012_corsika.012379.00001*.i3.bz2 -gcd /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/11058/01000-01999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz -t corsika -set 12379 -y 2012

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/11905/00000-00999/Level2_IC86.2012_corsika.011905.00001*.i3.bz2 -gcd /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/11058/01000-01999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz -t corsika -set 11905 -y 2012

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/11926/00000-00999/Level2_IC86.2012_corsika.011926.00001*.i3.bz2 -gcd /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/11058/01000-01999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz -t corsika -set 11926 -y 2012

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/11937/00000-00999/Level2_IC86.2012_corsika.011937.00001*.i3.bz2 -gcd /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/11058/01000-01999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz -t corsika -set 11937 -y 2012

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/12161/0000000-0000999/Level2_IC86.2012_corsika.012161.00001*.i3.bz2 -gcd /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/11058/01000-01999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz -t corsika -set 12161 -y 2012

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/12268/0000000-0000999/Level2_IC86.2012_corsika.012268.00002*.i3.bz2 -gcd /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/11058/01000-01999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz -t corsika -set 12268 -y 2012

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/10670/00000-00999/Level2_IC86.2012_corsika.010670.00001*.i3.bz2 -gcd /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/11058/01000-01999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz -t corsika -set 10670 -y 2012

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/11058/00000-00999/Level2_IC86.2012_corsika.011058.00001*.i3.bz2 -gcd /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/11058/01000-01999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz -t corsika -set 11058 -y 2012

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/11057/00000-00999/Level2_IC86.2012_corsika.011057.00001*.i3.bz2 -gcd /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/11058/01000-01999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz -t corsika -set 11057 -y 2012

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2016/filtered/level2/CORSIKA-in-ice/20787/0000000-0000999/Level2_IC86.2016_corsika.020787.000001.i3.zst -gcd /cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withStdNoise.i3.gz -t corsika -set 20787 -y 2016 

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2016/filtered/level2/CORSIKA-in-ice/20789/0000000-0000999/Level2_IC86.2016_corsika.020789.00001*.i3.zst -gcd /cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withStdNoise.i3.gz -t corsika -set 20789 -y 2016

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2016/filtered/level2/CORSIKA-in-ice/20848/0000000-0000999/Level2_IC86.2016_corsika.020848.00001*.i3.zst -gcd /cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withStdNoise.i3.gz -t corsika -set 20848 -y 2016 

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2016/filtered/level2/CORSIKA-in-ice/20849/0000000-0000999/Level2_IC86.2016_corsika.020849.00001*.i3.zst -gcd /cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withStdNoise.i3.gz -t corsika -set 20849 -y 2016 

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2016/filtered/level2/CORSIKA-in-ice/20852/0000000-0000999/Level2_IC86.2016_corsika.020852.0000*.i3.zst -gcd /cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withStdNoise.i3.gz -t corsika -set 20852 -y 2016 

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2016/filtered/level2/CORSIKA-in-ice/20904/0000000-0000999/Level2_IC86.2016_corsika.020904.0000*.i3.zst -gcd /cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withStdNoise.i3.gz -t corsika -set 20904 -y 2016 

#python /data/user/dpankova/double_pulse/Make_Images.py  -it 2 -o NuTau_Test_Images_BeforeEnd -i /data/ana/Cscd/StartingEvents/NuGen_new/NuTau/medium_energy/IC86_flasher_p1=0.3_p2=0.0/l2/1/l2_0000010*.i3.zst -gcd /data/user/dpankova/double_pulse/GeoCalibDetectorStatus_2013.56429_V1_Modified.i3.gz -t genie  

#python /data/user/dpankova/double_pulse/Make_Images.py  -it 1 -o Genie_Make_image_TEST -i /data/ana/Cscd/StartingEvents/NuGen_new/NuTau/medium_energy/IC86_flasher_p1=0.3_p2=0.0/l2/1/l2_00000001.i3.zst -gcd /data/user/dpankova/double_pulse/GeoCalibDetectorStatus_2013.56429_V1_Modified.i3.gz -t genie
