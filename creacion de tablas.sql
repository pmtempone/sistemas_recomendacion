--creacion de tabla de cantidad de seguidores por usuario

select id_usuario,count(id_usuario_seguido) as cant_seguidores
into usuarios_seguidores
from siguiendo
group by id_usuario;

--tabla de datos de restaurantes

select id_restaurante,localidad,cocina,precio,
	   case when telefono is not null then 1 else 0 end as telefono,
       precio, 
       case when char_length(rating_comida)>4 then 
       cast(substr(rating_comida,1,2) as numeric)/30 
       else cast(substr(rating_comida,1,1) as numeric)/30 end comida_oleo,
       case when char_length(rating_servicio)>4 then 
       cast(substr(rating_servicio,1,2) as numeric)/30 
       else cast(substr(rating_servicio,1,1) as numeric)/30 end servicio_oleo,
       case when char_length(rating_ambiente)>4 then 
       cast(substr(rating_ambiente,1,2) as numeric)/30 
       else cast(substr(rating_ambiente,1,1) as numeric)/30 end ambiente_oleo,
       fotos
into rest_campos
from restaurantes

-- base para entrenar rf

select a.*,b.edad,b.fecha_alta,b.genero,b.tipo, 
	   c.localidad,c.cocina,c.precio,c.telefono, c.comida_oleo,c.servicio_oleo,c.ambiente_oleo,
       c.fotos
into train_completo_v1
from
ratings_train a,usuarios b,rest_campos c
where a.id_usuario=b.id_usuario
and cast(a.id_restaurante as text)=c.id_restaurante

-- saco los clientes que comentaron 1 sola vez

select * 
into ratings_train_reducido
from ratings_train
where id_usuario in (select id_usuario from (select id_usuario,count(id_usuario)
from ratings_train
group by id_usuario
having count(id_usuario)>4) a)

