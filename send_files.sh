ls -l vivado-base/vivado-base.runs/impl_1/base_wrapper.bit -l
ls -l vivado-base/vivado-base.gen/sources_1/bd/base/hw_handoff/base.hwh -l
cp vivado-base/vivado-base.runs/impl_1/base_wrapper.bit generated_files/overlay.bit
cp vivado-base/vivado-base.gen/sources_1/bd/base/hw_handoff/base.hwh generated_files/overlay.hwh
scp generated_files/* ubuntu@192.168.0.11:/home/ubuntu/proyecto-grado/generated-files
