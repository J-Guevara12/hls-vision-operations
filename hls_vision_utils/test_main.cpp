int main_window_manager();
int main_read();
int main_write();

int main(){
    if (main_window_manager()!=0)
        return 1;

    if (main_read()!=0)
        return 1;

    if (main_write()!=0)
        return 1;

    return 0;
}
