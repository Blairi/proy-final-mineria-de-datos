import modules.phase1_2_universe as m1
import modules.phase3_preprocessing as m3
import modules.phase4_transforming as m4

RUTA_CSV      = './ENIGH_INEGI/viviendas.csv'
CVE_ESTADO    = '15'
NOMBRE_ESTADO = 'Estado de Mexico'
ATRIBUTOS     = ['num_cuarto', 'tam_loc', 'calent_sol', 'p_fractura', 'regadera', 'tenencia']

if __name__ == '__main__':

    # Fase 1 y 2
    df_estado = m1.ejecutar(RUTA_CSV, CVE_ESTADO, NOMBRE_ESTADO, ATRIBUTOS)

    # Fase 3
    df_work = m3.ejecutar(df_estado, ATRIBUTOS)
    
    # Fase 4
    X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler = m4.ejecutar(df_work)