--creacion de tabla de cantidad de seguidores por usuario

select id_usuario,count(id_usuario_seguido) as cant_seguidores
into usuarios_seguidores
from siguiendo
group by id_usuario;